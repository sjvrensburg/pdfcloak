"""Regression tests for the adversarial review fixes."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pymupdf as fitz
import pytest

from pdfcloak import PdfCloakError
from pdfcloak.cli import main
from pdfcloak.detectors import ClipCloakDetector
from pdfcloak.models import Span
from pdfcloak.redact import detect_on_pages


class _CountingDetector:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def detect(self, text, *, page, labels, threshold):
        self.calls += 1
        i = text.find("secret@example.com")
        return [Span(page, i, i + 18, "email", 2.0, "counting",
                     "secret@example.com")] if i >= 0 else []


def test_detector_cache_is_per_page(tmp_path):
    path = tmp_path / "dup.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 750), "mail secret@example.com now")
    doc.save(path)
    doc.close()
    det = _CountingDetector()
    with fitz.open(path) as doc:
        spans, _, _ = detect_on_pages(doc, [0, 1], [det], labels=[], threshold=0.5)
    assert [s.page for s in spans] == [0, 1]  # page 1 not lost to the cache


def test_chunk_straddling_entity_found():
    """An email cut in half by the 35k chunk boundary must still be found."""
    class Echo:
        pass

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = body["text"]
            i = text.find("jane@example.com")
            ents = ([{"label": "email", "text": "jane@example.com",
                      "start": i, "end": i + 16, "score": 0.9}] if i >= 0 else [])
            data = json.dumps({"entities": ents}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        det = ClipCloakDetector(f"http://127.0.0.1:{server.server_port}", "t")
        text = "x" * 34_995 + "jane@example.com" + "y" * 500
        spans = det.detect(text, page=0, labels=["email"], threshold=0.5)
        assert len(spans) == 1
        assert text[spans[0].start:spans[0].end] == "jane@example.com"
    finally:
        server.shutdown()


def test_missing_entities_key_fails_closed(monkeypatch):
    det = ClipCloakDetector("http://127.0.0.1:9", "t")
    monkeypatch.setattr(det, "_classify", lambda chunk, page, threshold: {})
    with pytest.raises(PdfCloakError, match="entities"):
        det.detect("hello", page=0, labels=[], threshold=0.5)


def test_batch_redact_into_input_dir_refused(sample_pdf, tmp_path, capsys):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(sample_pdf.read_bytes())
    assert main(["batch", "redact", str(folder), "-o", str(folder)]) == 2


def test_batch_recursive_mirrors_paths(sample_pdf, tmp_path, capsys):
    folder = tmp_path / "in"
    (folder / "x").mkdir(parents=True)
    (folder / "y").mkdir()
    for sub in ("x", "y"):
        (folder / sub / "same.pdf").write_bytes(sample_pdf.read_bytes())
    outdir = tmp_path / "out"
    rc = main(["batch", "redact", str(folder), "-o", str(outdir), "--recursive"])
    assert rc == 0
    assert (outdir / "x" / "same.pdf").exists()
    assert (outdir / "y" / "same.pdf").exists()


def test_dry_run_creates_preview(sample_pdf, rules_file, tmp_path):
    out, preview = tmp_path / "o.pdf", tmp_path / "p.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out), "--dry-run",
               "--preview", str(preview), "--rules", str(rules_file)])
    assert rc == 0 and preview.exists() and not out.exists()


def test_interactive_non_tty_refused(sample_pdf, rules_file, tmp_path):
    out = tmp_path / "o.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out), "--interactive",
               "--rules", str(rules_file)])
    assert rc == 2 and not out.exists()


def test_pdfcloak_lang_env(sample_pdf, capsys, monkeypatch):
    monkeypatch.setenv("PDFCLOAK_LANG", "de")
    assert main(["scan", str(sample_pdf)]) == 0
    assert "gefunden" in capsys.readouterr().err.lower()


def test_junk_prefix_pdf_opens(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "hello")
    raw = doc.tobytes()
    doc.close()
    path = tmp_path / "junk.pdf"
    path.write_bytes(b"% junk prefix line\n" + raw)
    assert main(["scan", str(path)]) == 0


def test_exclude_labels_disclosed_not_clean(sample_pdf, rules_file, tmp_path, capsys):
    out = tmp_path / "o.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out),
               "--exclude-labels", "email", "--rules", str(rules_file)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "verified clean" not in captured.err
    report = json.loads(captured.out)
    assert report["retained_excluded"]


def test_web_bad_content_length_gets_400():
    from pdfcloak.web import serve
    server = serve("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                          timeout=5)
        conn.putrequest("POST", "/scan")
        conn.putheader("Content-Length", "abc")
        conn.endheaders()
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()


def test_profile_labels_union(sample_pdf, capsys):
    rc = main(["scan", str(sample_pdf), "--profile", "hipaa",
               "--labels", "custom_label", "--detector", "regex"])
    assert rc == 0  # union must not error or drop the run
