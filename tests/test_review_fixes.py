"""Regression tests for the adversarial code review fixes."""
import pytest
import pymupdf as fitz
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from pdfcloak import PdfCloakError
from pdfcloak.cli import main
from pdfcloak.detectors import RegexDetector, _reconcile
from pdfcloak.pages import parse_page_spec
from pdfcloak.rules import Rule, load_rules_file


def test_bare_dash_rejected_and_open_range_clamps():
    with pytest.raises(PdfCloakError):
        parse_page_spec("-", 5)
    assert parse_page_spec("-99", 5) == [0, 1, 2, 3, 4]
    with pytest.raises(PdfCloakError):
        parse_page_spec("6", 5)


def test_rules_file_bad_types_are_clean_errors(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('rule = "email"\n', encoding="utf-8")
    with pytest.raises(PdfCloakError):
        load_rules_file(bad)
    bad.write_bytes(b"\xff\xfe[[rule]]")
    with pytest.raises(PdfCloakError):
        load_rules_file(bad)


def test_max_matches_fails_closed():
    det = RegexDetector([Rule("email", r"[\w.+-]+@[\w-]+\.[\w.]+")])
    text = " ".join(f"u{i}@x.example.org" for i in range(600))
    with pytest.raises(PdfCloakError, match="cap"):
        det.detect(text, page=0, labels=[], threshold=0.5)


def test_reconcile_prefers_nearest_duplicate():
    text = "José García met José García today"
    second = text.rindex("José")
    # Drifted offsets must map to the *second* occurrence, never the first.
    assert _reconcile(text, second + 20, second + 31, "José García") == (second, second + 11)
    assert _reconcile(text, 500, 510, "José García") is None


def _make_pdf(tmp_path, draw):
    path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    draw(c)
    c.save()
    return path


def test_split_line_email_detected_and_redacted(tmp_path, rules_file):
    def draw(c):
        c.drawString(72, 750, "Contact jane.doe@")
        c.drawString(72, 730, "example.com for details.")
        c.showPage()
    path = _make_pdf(tmp_path, draw)
    out = tmp_path / "out.pdf"
    assert main(["redact", str(path), "-o", str(out), "--rules", str(rules_file)]) == 0
    with fitz.open(out) as doc:
        text = doc[0].get_text()
    assert "jane.doe@" not in text and "example.com" not in text


def test_corrupt_and_encrypted_are_clean_errors(tmp_path):
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf at all")
    assert main(["scan", str(junk)]) == 2
    enc = tmp_path / "enc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "secret jane@example.com")
    doc.save(enc, encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="pw123", owner_pw="pw123")
    doc.close()
    assert main(["scan", str(enc)]) == 2
    assert main(["scan", str(enc), "--password", "pw123"]) == 0
    assert main(["scan", str(enc), "--password", "wrong"]) == 2


def test_threshold_and_labels_validation(sample_pdf):
    with pytest.raises(SystemExit):
        main(["scan", str(sample_pdf), "--threshold", "nan"])
    with pytest.raises(SystemExit):
        main(["scan", str(sample_pdf), "--threshold", "1.5"])
    assert main(["scan", str(sample_pdf), "--labels", ",,"]) == 2
    assert main(["scan", str(sample_pdf), "--detector", "regex",
                 "--detector", "regex"]) == 0


def test_scrub_removes_annots_widgets_links_toc_on_all_pages(tmp_path, rules_file):
    path = tmp_path / "carriers.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 750), "Reach jane.doe@example.com now")
    p2 = doc.new_page()
    p2.insert_text((72, 750), "Visible text stays on page two")
    p2.add_text_annot((100, 700), "secret admin@corp.example.org")
    p2.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 720, 200, 735),
                    "uri": "mailto:admin@corp.example.org"})
    widget = fitz.Widget()
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    widget.field_name = "email"
    widget.field_value = "admin@corp.example.org"
    widget.rect = fitz.Rect(72, 650, 300, 670)
    p2.add_widget(widget)
    doc.set_toc([[1, "admin@corp.example.org bookmark", 1]])
    doc.save(path)
    doc.close()

    out = tmp_path / "out.pdf"
    # Only page 1 is in scope for text, but invisible carriers go everywhere.
    assert main(["redact", str(path), "-o", str(out), "--pages", "1",
                 "--rules", str(rules_file)]) == 0
    with fitz.open(out) as doc:
        assert doc.get_toc() == []
        for page in doc:
            assert list(page.annots() or []) == []
            assert list(page.widgets() or []) == []
            assert page.get_links() == []
        assert "Visible text stays on page two" in doc[1].get_text()
    assert b"admin@corp.example.org" not in out.read_bytes()


def test_scanned_pages_are_not_verified_clean(tmp_path):
    path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(10, 10, 100, 100), pixmap=pix)
    doc.save(path)
    doc.close()
    assert main(["verify", str(path)]) == 3
    out = tmp_path / "out.pdf"
    assert main(["redact", str(path), "-o", str(out)]) == 3
    assert out.exists()
    assert main(["redact", str(path), "-o", str(out), "--allow-scanned"]) == 0


def test_onnx_detector_chunks_thresholds_and_label_vocab():
    import numpy as np

    from pdfcloak.detectors import OnnxDetector

    class Enc:
        def __init__(self, text):
            self.tokens = text.split(" ")
            self.ids = list(range(len(self.tokens)))
            self.attention_mask = [1] * len(self.tokens)
            self.type_ids = [0] * len(self.tokens)
            self.offsets = []
            pos = 0
            for tok in self.tokens:
                self.offsets.append((pos, pos + len(tok)))
                pos += len(tok) + 1

    class FakeTokenizer:
        def encode(self, text):
            return Enc(text)

    class Input:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [Input("input_ids"), Input("attention_mask")]

        def run(self, _outputs, feed):
            ids = feed["input_ids"][0]
            enc_tokens = CURRENT[0]
            logits = np.full((1, len(ids), 2), -10.0)
            for j, tok_id in enumerate(ids):
                logits[0, j, 0] = 10.0
                if enc_tokens[int(tok_id)] == "Jane":
                    logits[0, j] = [-10.0, 10.0]
            return [logits]

    CURRENT = [[]]
    det = object.__new__(OnnxDetector)
    det.session = FakeSession()
    det.tokenizer = FakeTokenizer()
    det.id2label = {0: "O", 1: "B-PER"}
    det.input_names = ["input_ids", "attention_mask"]

    text = " ".join(["word"] * 600 + ["Jane"])
    CURRENT[0] = text.split(" ")
    spans = det.detect(text, page=0, labels=["PER"], threshold=0.55)
    assert [s.text for s in spans] == ["Jane"]  # beyond token 512: chunked
    assert spans[0].score > 0.9  # real softmax score, not hardcoded 1.0
    assert det.detect(text, page=0, labels=["PER"], threshold=1.0) == []
    with pytest.raises(PdfCloakError, match="do not overlap"):
        det.detect(text, page=0, labels=["name"], threshold=0.55)


def test_wheel_manifest_covers_all_vendored_sources():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    included = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
                   ["force-include"].keys())
    vendor = root / "vendor" / "clipcloak-server"
    on_disk = {str(p.relative_to(root)) for p in vendor.rglob("*")
               if p.is_file() and "target" not in p.parts
               and "__pycache__" not in p.parts}
    assert on_disk <= included, f"not bundled in wheel: {sorted(on_disk - included)}"
