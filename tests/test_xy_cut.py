"""XY-Cut++-style reading order tests."""
import pymupdf as fitz
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from pdfcloak.extract import BlockItem, LineItem, extract_page, order_blocks, xy_cut_order


def _block(text, x0, y0, w, h):
    rect = fitz.Rect(x0, y0, x0 + w, y0 + h)
    line = LineItem(bbox=rect, chars=[(c, rect) for c in text])
    return BlockItem(bbox=fitz.Rect(rect), lines=[line])


def _block_texts(blocks):
    return [b.lines[0].text for b in order_blocks(blocks)]


def _item(text, x0, y0, x1=None, height=10):
    x1 = x1 if x1 is not None else x0 + max(40, len(text) * 5)
    rect = fitz.Rect(x0, y0, x1, y0 + height)
    return LineItem(bbox=rect, chars=[(c, rect) for c in text])


def _texts(items):
    return [i.text for i in xy_cut_order(items)]


def test_two_columns_left_before_right():
    left = [_item("L1", 50, 100), _item("L2", 50, 130), _item("L3", 50, 160)]
    right = [_item("R1", 320, 100), _item("R2", 320, 130), _item("R3", 320, 160)]
    # Deliberately scrambled stream order.
    assert _texts(right + left) == ["L1", "L2", "L3", "R1", "R2", "R3"]


def test_single_column_top_to_bottom():
    items = [_item("third", 50, 160), _item("first", 50, 100), _item("second", 50, 130)]
    assert _texts(items) == ["first", "second", "third"]


def test_spanning_title_masked_and_first():
    title = _item("TITLE SPANNING BOTH COLUMNS", 50, 60, x1=560)
    left = [_item("L1", 50, 100), _item("L2", 50, 130)]
    right = [_item("R1", 320, 100), _item("R2", 320, 130)]
    result = _texts(left + right + [title])
    assert result[0] == "TITLE SPANNING BOTH COLUMNS"
    assert result[1:] == ["L1", "L2", "R1", "R2"]


def test_spanning_footer_masked_and_last():
    footer = _item("FOOTER ACROSS THE WHOLE PAGE WIDTH", 50, 700, x1=560)
    left = [_item("L1", 50, 100), _item("L2", 50, 130)]
    right = [_item("R1", 320, 100), _item("R2", 320, 130)]
    result = _texts([footer] + right + left)
    assert result == ["L1", "L2", "R1", "R2", "FOOTER ACROSS THE WHOLE PAGE WIDTH"]


def test_mid_document_spanning_heading():
    heading = _item("SECTION HEADING ACROSS COLUMNS", 50, 300, x1=560)
    above_l = [_item("AL1", 50, 100), _item("AL2", 50, 130)]
    above_r = [_item("AR1", 320, 100), _item("AR2", 320, 130)]
    below_l = [_item("BL1", 50, 340), _item("BL2", 50, 370)]
    below_r = [_item("BR1", 320, 340), _item("BR2", 320, 370)]
    result = _texts(below_r + above_l + [heading] + below_l + above_r)
    assert result == ["AL1", "AL2", "AR1", "AR2",
                      "SECTION HEADING ACROSS COLUMNS",
                      "BL1", "BL2", "BR1", "BR2"]


def test_projection_gutter_tolerates_crosser():
    left = _block("LEFT", 50, 100, 200, 300)
    right = _block("RIGHT", 320, 100, 200, 300)
    crosser = _block("EQ", 200, 250, 180, 20)  # pokes across the gutter
    result = _block_texts([right, crosser, left])
    # No clean gutter exists (the equation straddles it); the projection
    # profile still finds the columns and reads the crosser between them.
    assert result[0] == "LEFT"
    assert result.index("RIGHT") > result.index("LEFT")
    assert set(result) == {"LEFT", "RIGHT", "EQ"}


def test_margin_note_masked_and_reinserted_beside_body():
    main1 = _block("MAIN1", 100, 100, 300, 200)
    main2 = _block("MAIN2", 100, 320, 300, 100)
    note = _block("NOTE", 430, 150, 70, 60)  # narrow, beside MAIN1
    result = _block_texts([main2, note, main1])
    assert result == ["MAIN1", "NOTE", "MAIN2"]


def test_two_column_pdf_extracts_columnwise(tmp_path):
    path = tmp_path / "cols.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    # Draw the right column first: stream order must not matter.
    for i, y in enumerate((700, 670, 640)):
        c.drawString(320, y, f"Right line {i}")
    for i, y in enumerate((700, 670, 640)):
        c.drawString(60, y, f"Left line {i}")
    c.showPage()
    c.save()
    with fitz.open(path) as doc:
        text = extract_page(doc[0], 0).text
    assert text.index("Left line 2") < text.index("Right line 0")
    assert text == ("Left line 0\nLeft line 1\nLeft line 2\n"
                    "Right line 0\nRight line 1\nRight line 2")
