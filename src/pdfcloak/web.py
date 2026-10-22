"""Local web UI stepping stone: `pdfcloak serve`.

Standard-library only. Binds loopback by default and refuses other hosts
unless PDFCLOAK_ALLOW_REMOTE_SERVE=1, because it accepts PDF uploads and
has no authentication. It reuses the regex pipeline (built-in rules);
matches are shown masked, never as raw PII. Uploaded bytes are processed
in a temporary directory and deleted with the request.
"""
from __future__ import annotations

import html
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import PdfCloakError
from .detectors import RegexDetector
from .models import masked_text
from .redact import apply_redactions, detect_on_pages, save_safely, scrub_document
from .rules import built_in_rules

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pdfcloak — local PDF redaction</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#16181d;
background:#fff;max-width:44rem;margin:2.5rem auto;padding:0 1rem;line-height:1.5}
h1{font-family:Charter,Georgia,"Times New Roman",serif;font-size:2rem;
font-weight:700;letter-spacing:-.01em;margin-bottom:.25rem}
table{border-collapse:collapse;margin-top:1rem}
td,th{border:1px solid #6b7280;padding:.35rem .6rem;text-align:left}
th{background:#16181d;color:#fff}
fieldset{margin:1rem 0}
:focus-visible{outline:3px solid #005fcc;outline-offset:2px}
button{padding:.55rem 1.1rem;background:#16181d;color:#fff;border:0;
font-weight:600;cursor:pointer}
</style>
</head><body>
<h1>pdfcloak</h1>
<p>Everything runs on this machine. No PDF is uploaded anywhere else.</p>
<form method="post" action="/scan" enctype="multipart/form-data">
<label for="file">PDF file</label>
<input id="file" name="file" type="file" accept="application/pdf" required>
<fieldset><legend>Action</legend>
<label><input type="radio" name="action" value="scan" checked> Scan only</label>
<label><input type="radio" name="action" value="redact"> Redact and download</label>
</fieldset>
<button type="submit">Run</button>
</form>
{result}
</body></html>"""


MAX_WEB_PAGES = 200  # an upload must not make the UI scan unbounded work


def _check_page_count(count: int) -> None:
    if count > MAX_WEB_PAGES:
        raise PdfCloakError(
            f"uploaded PDF has {count} pages; the web UI accepts at most "
            f"{MAX_WEB_PAGES} (use the CLI for larger files)")


def _detector() -> RegexDetector:
    return RegexDetector(built_in_rules())


def scan_bytes(data: bytes) -> list[dict]:
    """Scan uploaded PDF bytes; return masked match dicts."""
    import pymupdf as fitz

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "upload.pdf"
        src.write_bytes(data)
        if data[:5] != b"%PDF-":
            raise PdfCloakError("uploaded file is not a PDF")
        with fitz.open(src) as doc:
            _check_page_count(doc.page_count)
            spans, page_texts, scanned = detect_on_pages(
                doc, list(range(doc.page_count)), [_detector()],
                labels=[], threshold=0.55)
            by_page = {pt.page: pt for pt in page_texts}
            out = []
            for s in spans:
                text = s.text or by_page[s.page].text[s.start:s.end]
                out.append({"page": s.page + 1, "label": s.label,
                            "masked": masked_text(text)})
            if scanned:
                out.append({"page": scanned[0] + 1, "label": "unchecked",
                            "masked": f"{len(scanned)} page(s) have no text layer"})
            return out


def redact_bytes(data: bytes) -> bytes:
    """Redact uploaded PDF bytes; return the redacted PDF bytes."""
    import pymupdf as fitz

    with tempfile.TemporaryDirectory() as tmp:
        src, dest = Path(tmp) / "upload.pdf", Path(tmp) / "out.pdf"
        src.write_bytes(data)
        if data[:5] != b"%PDF-":
            raise PdfCloakError("uploaded file is not a PDF")
        with fitz.open(src) as doc:
            _check_page_count(doc.page_count)
            pages = list(range(doc.page_count))
            spans, page_texts, _ = detect_on_pages(
                doc, pages, [_detector()], labels=[], threshold=0.55)
            by_page = {pt.page: pt for pt in page_texts}
            for s in spans:
                if not s.text:
                    object.__setattr__(s, "text", by_page[s.page].text[s.start:s.end])
            apply_redactions(doc, spans, by_page, replacement="black")
            scrub_document(doc, pages)
            save_safely(doc, dest)
        return dest.read_bytes()


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Minimal multipart/form-data parser for a single small form.

    Splits on the full delimiter, strips exactly the one CRLF that precedes
    each delimiter (payload bytes ending in '-' or CRLF survive), and parses
    quoted disposition parameters so filenames containing ';' work.
    """
    import re

    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip('"')
    if not boundary:
        raise PdfCloakError("expected multipart/form-data upload")
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    delimiter = b"--" + boundary.encode()
    for chunk in body.split(delimiter):
        if chunk.startswith(b"--"):
            break  # closing delimiter
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if b"\r\n\r\n" not in chunk:
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        params: dict[str, str] = {}
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for m in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', line):
                    params[m.group(1)] = m.group(2).replace('\\"', '"')
        name = params.get("name")
        if not name:
            continue
        if params.get("filename"):
            files[name] = payload
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files

def make_handler():
    class Handler(BaseHTTPRequestHandler):
        timeout = 60

        def _check_same_site(self) -> None:
            """Block cross-site form posts (drive-by localhost attacks)."""
            import urllib.parse
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host and host not in ("127.0.0.1", "localhost", "::1"):
                raise PdfCloakError("unexpected Host header")
            origin = self.headers.get("Origin")
            if origin:
                ohost = (urllib.parse.urlparse(origin).hostname or "")
                if ohost not in ("127.0.0.1", "localhost", "::1"):
                    raise PdfCloakError("cross-site form posts are not accepted")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, _PAGE.replace("{result}", "").encode(),
                       "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._check_same_site()
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    raise PdfCloakError("invalid Content-Length header") from None
                if length < 0 or length > 50 * 1024 * 1024:
                    raise PdfCloakError("upload too large (max 50 MB)")
                fields, files = _parse_multipart(
                    self.rfile.read(length), self.headers.get("Content-Type", ""))
                data = files.get("file")
                if not data:
                    raise PdfCloakError("no PDF file in upload (field 'file')")
                if fields.get("action") == "redact" or self.path == "/redact":
                    self._send(200, redact_bytes(data), "application/pdf")
                    return
                rows = scan_bytes(data)
                table = "<p>No matches found.</p>" if not rows else (
                    "<table><caption>Matches (masked)</caption>"
                    "<tr><th scope='col'>Page</th><th scope='col'>Label</th>"
                    "<th scope='col'>Match</th></tr>" + "".join(
                        f"<tr><td>{r['page']}</td><td>{html.escape(r['label'])}</td>"
                        f"<td>{html.escape(r['masked'])}</td></tr>" for r in rows)
                    + "</table>")
                self._send(200, _PAGE.replace("{result}", table).encode(),
                           "text/html; charset=utf-8")
            except PdfCloakError as exc:
                msg = f"<p role='alert'>Error: {html.escape(str(exc))}</p>"
                self._send(400, _PAGE.replace("{result}", msg).encode(),
                           "text/html; charset=utf-8")
            except Exception as exc:  # never drop the connection silently
                msg = (f"<p role='alert'>Error: "
                       f"{html.escape(exc.__class__.__name__)}</p>")
                try:
                    self._send(400, _PAGE.replace("{result}", msg).encode(),
                               "text/html; charset=utf-8")
                except OSError:
                    pass

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create (but do not start) the UI server; loopback unless opted in."""
    if host not in ("127.0.0.1", "localhost", "::1") \
            and os.environ.get("PDFCLOAK_ALLOW_REMOTE_SERVE") != "1":
        raise PdfCloakError(
            f"refusing to bind {host}: the UI has no authentication. "
            "Set PDFCLOAK_ALLOW_REMOTE_SERVE=1 if you accept that risk.")
    server = ThreadingHTTPServer((host, port), make_handler())
    server.daemon_threads = True
    return server
