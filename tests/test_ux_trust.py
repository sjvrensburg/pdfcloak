"""Tests for the UX/trust hardening pass (see audit)."""
import json
import os
import stat

import pymupdf as fitz
import pytest

from pdfcloak import PdfCloakError
from pdfcloak.cli import _secret, main
from pdfcloak.detectors import ClipCloakDetector, RegexDetector, _text_chunks
from pdfcloak.models import Span, masked_text
from pdfcloak.rules import built_in_rules


def _scan_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_custom_regex_adds_to_builtins_not_replaces(sample_pdf, capsys):
    # A --regex for an ID must not turn off built-in email detection.
    rc = main(["scan", str(sample_pdf), "--regex", r"sa_id:\b\d{13}\b"])
    assert rc == 0
    report = _scan_json(capsys)
    labels = {m["label"] for m in report["matches"]}
    assert "email" in labels and "sa_id" in labels


def test_no_builtins_disables_email(sample_pdf, capsys):
    rc = main(["scan", str(sample_pdf), "--regex", r"sa_id:\b\d{13}\b",
               "--no-builtins"])
    assert rc == 0
    report = _scan_json(capsys)
    assert {m["label"] for m in report["matches"]} == {"sa_id"}


def test_unicode_email_builtin():
    det = RegexDetector(built_in_rules())
    spans = det.detect("write to josé@exemplo.example.org today",
                       page=0, labels=[], threshold=0.5)
    assert [s.label for s in spans] == ["email"]


def test_masked_text_shapes():
    assert masked_text("jane.doe@example.com") == "j***@e***.com"
    assert masked_text("secret") == "***"        # short values fully masked
    assert masked_text("a@b.co") == "***"        # reconstructable -> masked
    assert masked_text("longer-value") == "l***e"
    assert masked_text("abc") == "***"
    assert masked_text("") == ""


def test_report_entries_hide_hash_and_score_rules(sample_pdf, capsys):
    assert main(["scan", str(sample_pdf)]) == 0
    report = _scan_json(capsys)
    assert report["schema_version"] == 1
    assert report["counts_by_label"]["email"] == 2
    entry = report["matches"][0]
    assert "sha256" not in entry  # unsalted hash of low-entropy PII
    assert entry["score"] is None and entry["priority"] == "rule"
    assert entry["masked_text"] and "example" not in entry["masked_text"]


def test_report_digest_only_when_keyed(sample_pdf, capsys, monkeypatch):
    monkeypatch.setenv("PDFCLOAK_REPORT_HMAC_KEY", "test-key")
    assert main(["scan", str(sample_pdf)]) == 0
    entry = _scan_json(capsys)["matches"][0]
    assert entry["digest"] and "sha256" not in entry


def test_scan_summary_and_fail_on_findings(sample_pdf, capsys):
    assert main(["scan", str(sample_pdf)]) == 0
    assert "found: 2 match(es)" in capsys.readouterr().err
    assert main(["scan", str(sample_pdf), "--fail-on-findings"]) == 3
    capsys.readouterr()
    assert main(["scan", str(sample_pdf), "--quiet"]) == 0
    assert capsys.readouterr().err == ""


def test_scan_pagination(sample_pdf, capsys):
    assert main(["scan", str(sample_pdf), "--limit", "1"]) == 0
    report = _scan_json(capsys)
    assert report["count"] == 2 and report["returned"] == 1


def test_non_pdf_is_clean_error(tmp_path, capsys):
    junk = tmp_path / "notes.pdf"
    junk.write_text("jane.doe@example.com", encoding="utf-8")
    assert main(["scan", str(junk)]) == 2
    assert "not a PDF" in capsys.readouterr().err


def test_blank_page_is_unchecked_not_clean(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    assert main(["verify", str(path)]) == 3


def test_password_from_env(tmp_path, monkeypatch):
    enc = tmp_path / "enc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "secret jane@example.com")
    doc.save(enc, encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="pw123", owner_pw="pw123")
    doc.close()
    monkeypatch.setenv("PDFCLOAK_PASSWORD", "pw123")
    assert main(["scan", str(enc)]) == 0


def test_secret_prefers_env_over_nothing(monkeypatch):
    monkeypatch.delenv("PDFCLOAK_TOKEN", raising=False)
    assert _secret(None, None, "PDFCLOAK_TOKEN", "--token") is None
    monkeypatch.setenv("PDFCLOAK_TOKEN", "from-env")
    assert _secret(None, None, "PDFCLOAK_TOKEN", "--token") == "from-env"


def test_sidecar_url_validation():
    with pytest.raises(PdfCloakError):
        ClipCloakDetector("http://example.com:8731", "tok")
    with pytest.raises(PdfCloakError):
        ClipCloakDetector("https://example.com:8731", "tok")
    with pytest.raises(PdfCloakError):
        ClipCloakDetector("ftp://127.0.0.1", None)
    # Loopback is the normal local sidecar and is always fine.
    ClipCloakDetector("http://127.0.0.1:8731", "tok")


def test_text_chunks_reconstruct():
    text = "héllo wörld " * 5000
    chunks = _text_chunks(text, max_bytes=10_000)
    assert len(chunks) > 1
    assert all(len(c.encode()) <= 10_500 for _, c in chunks)
    # Overlapping chunks: every character offset is covered at least once.
    covered = set()
    for off, chunk in chunks:
        covered.update(range(off, off + len(chunk)))
    assert covered == set(range(len(text)))
    assert chunks[1][0] < len(chunks[0][1])  # overlap, not a hard cut


def test_malformed_sidecar_entity_fails_closed():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = json.dumps({"entities": [
                {"label": "name", "text": "Nobody Here",
                 "start": 9999, "end": 10010, "score": 0.9}]}).encode()
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
        with pytest.raises(PdfCloakError, match="malformed"):
            det.detect("hello world", page=0, labels=["name"], threshold=0.5)
    finally:
        server.shutdown()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode semantics")
def test_atomic_save_permissions_and_overwrite(sample_pdf, rules_file, tmp_path):
    out = tmp_path / "out.pdf"
    assert main(["redact", str(sample_pdf), "-o", str(out),
                 "--rules", str(rules_file)]) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    # Second run to the same path replaces atomically (existing contract).
    assert main(["redact", str(sample_pdf), "-o", str(out),
                 "--rules", str(rules_file)]) == 0


@pytest.mark.skipif(os.name == "nt", reason="symlink needs privileges on Windows")
def test_output_symlink_refused(sample_pdf, rules_file, tmp_path):
    target = tmp_path / "target.pdf"
    target.write_bytes(b"x")
    link = tmp_path / "link.pdf"
    link.symlink_to(target)
    assert main(["redact", str(sample_pdf), "-o", str(link),
                 "--rules", str(rules_file)]) == 2
    assert target.read_bytes() == b"x"


def test_scrub_report_and_dry_run_preview(sample_pdf, rules_file, tmp_path, capsys):
    out = tmp_path / "out.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out), "--dry-run",
               "--rules", str(rules_file)])
    assert rc == 0 and not out.exists()
    report = _scan_json(capsys)
    assert report["dry_run"] is True and "scrub_preview" in report


def test_preview_and_exclude_labels(sample_pdf, rules_file, tmp_path):
    out, preview = tmp_path / "out.pdf", tmp_path / "preview.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out), "--preview", str(preview),
               "--exclude-labels", "email", "--rules", str(rules_file)])
    assert rc == 0 and preview.exists()
    with fitz.open(out) as doc:
        text = "\n".join(p.get_text() for p in doc)
    assert "jane.doe@example.com" in text  # excluded label left in place
    assert "8001015009087" not in text     # sa_id still redacted


def test_survivors_are_quarantined(sample_pdf, rules_file, tmp_path, monkeypatch):
    import pdfcloak.redact as redact_mod

    survivor = Span(0, 0, 5, "email", 1.0, "verify", "jane.")
    monkeypatch.setattr(redact_mod, "verify_output",
                        lambda *a, **k: ([survivor], []))
    out = tmp_path / "out.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out),
               "--rules", str(rules_file)])
    assert rc == 3
    assert not out.exists()
    assert (tmp_path / "out.pdf.unsafe").exists()


def test_sidecar_doctor_gate_missing(capsys, monkeypatch):
    import pdfcloak.cli as cli
    monkeypatch.setattr(cli, "doctor_checks", lambda: [
        ("sidecar binary", False, "run: pdfcloak sidecar build"),
        ("cargo (only needed to build)", True, "install Rust"),
        ("onnxruntime library", False, "set ORT_DYLIB_PATH"),
        ("model weights cached", False, "first run downloads")])
    assert main(["sidecar", "doctor"]) == 3
    out = capsys.readouterr().out
    assert "MISS sidecar binary" in out and "fix:" in out


def test_sidecar_doctor_gate_ready(capsys, monkeypatch):
    import pdfcloak.cli as cli
    monkeypatch.setattr(cli, "doctor_checks", lambda: [
        ("sidecar binary", True, ""), ("cargo (only needed to build)", False, ""),
        ("onnxruntime library", True, ""), ("model weights cached", True, "")])
    assert main(["sidecar", "doctor"]) == 0
    assert "ok   sidecar binary" in capsys.readouterr().out


def test_max_pages_limit(sample_pdf):
    assert main(["scan", str(sample_pdf), "--max-pages", "2"]) == 2
    assert main(["scan", str(sample_pdf), "--max-pages", "3"]) == 0


def test_identity_404_fails_closed():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                data = b'{"ok":true}'
                self.send_response(200)
            else:
                data = b'{"error":"not found"}'
                self.send_response(404)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        det = ClipCloakDetector(f"http://127.0.0.1:{server.server_port}", "t")
        assert det.health() is True
        # A squatter answering /health but not /identity is not trusted.
        assert det.identity() is False
    finally:
        server.shutdown()


def test_onnx_missing_model_dir_is_clean_error(tmp_path):
    from pdfcloak.detectors import OnnxDetector
    with pytest.raises(PdfCloakError):
        OnnxDetector(tmp_path)


def test_doctor_checks_rows():
    from pdfcloak.sidecar import doctor_checks
    rows = doctor_checks()
    assert [r[0] for r in rows][0] == "sidecar binary"
    assert all(isinstance(r[1], bool) and r[2] for r in rows)


def test_scrub_counts_and_lang(tmp_path):
    from pdfcloak.redact import scrub_document
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "hello jane@example.com")
    doc.set_toc([[1, "secret bookmark", 1]])
    doc.save(path)
    doc.close()
    with fitz.open(path) as doc:
        counts = scrub_document(doc, [0], lang="af")
        assert counts["toc_entries"] == 1
        assert doc.get_toc() == []
        assert doc.metadata["title"] == "Redacted document"
        _kind, value = doc.xref_get_key(doc.pdf_catalog(), "Lang")
        assert value == "af"
