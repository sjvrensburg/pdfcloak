import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@pytest.fixture()
def sample_pdf(tmp_path):
    """3-page born-digital PDF with known PII at known places."""
    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setAuthor("Jane Doe")
    c.drawString(72, 750, "Contact Jane Doe at jane.doe@example.com for details.")
    c.drawString(72, 720, "Her phone is +27 82 555 1234 and ID 8001015009087.")
    c.showPage()
    c.drawString(72, 750, "Nothing sensitive on page two.")
    c.showPage()
    c.drawString(72, 750, "Page three: admin@corp.example.org handles billing.")
    c.showPage()
    c.save()
    return path


@pytest.fixture()
def rules_file(tmp_path):
    p = tmp_path / "rules.toml"
    p.write_text(
        '[[rule]]\nname = "email"\n'
        'pattern = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}"\n'
        '[[rule]]\nname = "sa_id"\npattern = "\\\\b\\\\d{13}\\\\b"\n',
        encoding="utf-8")
    return p
