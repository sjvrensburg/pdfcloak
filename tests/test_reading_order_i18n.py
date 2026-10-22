"""RTL/vertical reading-order and i18n tests."""
import pymupdf as fitz

from pdfcloak.extract import LineItem, PageText, extract_page, logical_chars, xy_cut_order


def _chars(text, *, step=10, start=0, y=100, vertical=False):
    out = []
    for i, c in enumerate(text):
        if vertical:
            rect = fitz.Rect(50, start + i * step, 60, start + (i + 1) * step)
        else:
            rect = fitz.Rect(start + i * step, y, start + (i + 1) * step, y + 10)
        out.append((c, rect))
    return out


def test_rtl_visual_order_reversed_to_logical():
    # Stored left-to-right (visual) but direction is RTL: read right-to-left.
    chars = _chars("abc", step=10)
    result = logical_chars(chars, (-1.0, 0.0))
    assert "".join(c for c, _ in result) == "cba"


def test_rtl_logical_order_kept():
    # Stream order already runs right-to-left (decreasing x): logical order
    # is the stream order, so the text reads "cba" and stays untouched.
    chars = list(reversed(_chars("abc", step=10)))
    result = logical_chars(chars, (-1.0, 0.0))
    assert "".join(c for c, _ in result) == "cba"


def test_ltr_order_untouched():
    chars = _chars("abc")
    assert logical_chars(chars, (1.0, 0.0)) == chars


def test_vertical_columns_right_to_left():
    def col(text, x):
        rect = fitz.Rect(x, 100, x + 12, 200)
        return LineItem(bbox=rect, chars=[(c, rect) for c in text],
                        direction=(0.0, -1.0), wmode=1)
    right, left = col("RIGHT", 300), col("LEFT", 100)
    result = xy_cut_order([left, right])
    assert [li.text for li in result] == ["RIGHT", "LEFT"]


def test_rtl_rects_merge_into_one():
    pt = PageText(page=0)
    # RTL span: char boxes step leftwards.
    pt.rects = [fitz.Rect(100 - i * 10, 100, 110 - i * 10, 112) for i in range(4)]
    rects = pt.rects_for_span(0, 4)
    assert len(rects) == 1
    assert rects[0].x0 == 70 and rects[0].x1 == 110


def test_rotated_text_extracts_contiguous(tmp_path):
    path = tmp_path / "rot.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "Hello World", rotate=90)
    doc.save(path)
    doc.close()
    with fitz.open(path) as doc:
        text = extract_page(doc[0], 0).text
    assert "Hello World" in text


# --- i18n -----------------------------------------------------------------

from pdfcloak.i18n import translate  # noqa: E402


def test_translate_catalogues_and_fallback():
    assert "match" in translate("en", "found", count=2, detail="email=2")
    assert "treffer" in translate("af", "found", count=2, detail="email=2").lower()
    assert "Treffer" in translate("de", "found", count=2, detail="email=2")
    # Unknown language falls back to English, never crashes.
    assert "match" in translate("xx", "found", count=1, detail="email=1")
    assert translate("af", "clean") != translate("en", "clean")
