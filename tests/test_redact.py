import pymupdf as fitz

from pdfcloak.cli import main


def _all_text(path):
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def test_scan_finds_email_and_id(sample_pdf, rules_file, capsys):
    rc = main(["scan", str(sample_pdf), "--rules", str(rules_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "jane.doe@example.com" not in out  # hashed by default
    assert '"count": 3' in out  # 2 emails + 1 SA ID across doc


def test_scan_page_scope(sample_pdf, rules_file, capsys):
    rc = main(["scan", str(sample_pdf), "--pages", "2", "--rules", str(rules_file)])
    assert rc == 0
    assert '"count": 0' in capsys.readouterr().out


def test_redact_removes_text_and_metadata(sample_pdf, rules_file, tmp_path):
    out = tmp_path / "out.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out),
               "--rules", str(rules_file)])
    assert rc == 0
    text = _all_text(out)
    assert "jane.doe@example.com" not in text
    assert "admin@corp.example.org" not in text
    assert "8001015009087" not in text
    assert "Nothing sensitive on page two." in text
    raw = out.read_bytes()
    assert b"jane.doe@example.com" not in raw
    with fitz.open(out) as doc:
        assert not doc.metadata.get("author")


def test_redact_page_scope_leaves_others(sample_pdf, rules_file, tmp_path):
    out = tmp_path / "out.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out),
               "--pages", "1", "--rules", str(rules_file)])
    assert rc == 0
    text = _all_text(out)
    assert "jane.doe@example.com" not in text
    # Page 3 was out of scope and must be untouched.
    assert "admin@corp.example.org" in text


def test_verify_detects_unredacted(sample_pdf, rules_file):
    rc = main(["verify", str(sample_pdf), "--rules", str(rules_file)])
    assert rc == 3


def test_verify_output_separates_survivors_from_new_hits(sample_pdf, rules_file, tmp_path):
    from pdfcloak.models import Span
    from pdfcloak.redact import verify_output

    class AtDetector:
        name = "stub"

        def detect(self, text, *, page, labels, threshold):
            import re
            m = re.search(r"(?<!\w)at(?!\w)", text)
            return [Span(page, m.start(), m.end(), "name", 0.9, "stub", "at")] if m else []

    out = tmp_path / "out.pdf"
    assert main(["redact", str(sample_pdf), "-o", str(out),
                 "--rules", str(rules_file)]) == 0
    survivors, new_hits = verify_output(
        out, [0], [AtDetector()], labels=[], threshold=0.5,
        original_texts=["jane.doe@example.com"])
    assert survivors == []
    assert [s.text for s in new_hits] == ["at"]


def test_dry_run_writes_nothing(sample_pdf, rules_file, tmp_path):
    out = tmp_path / "out.pdf"
    rc = main(["redact", str(sample_pdf), "-o", str(out), "--dry-run",
               "--rules", str(rules_file)])
    assert rc == 0
    assert not out.exists()
