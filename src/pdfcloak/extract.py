"""Extract a page as one string plus a char-offset -> rectangle map.

The string is *constructed*: lines are joined with '\\n' and blocks with an
extra '\\n'. Separator characters get a None rectangle. Detectors run on this
string; spans are mapped back to per-character boxes from get_text('rawdict').
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf as fitz


@dataclass
class PageText:
    page: int  # 0-based
    text: str = ""
    rects: list[fitz.Rect | None] = field(default_factory=list)
    has_images: bool = False
    char_count: int = 0

    def rects_for_span(self, start: int, end: int) -> list[fitz.Rect]:
        """Merge char boxes of span [start, end) into one rect per line."""
        merged: list[fitz.Rect] = []
        for i in range(max(0, start), min(end, len(self.rects))):
            rect = self.rects[i]
            if rect is None or rect.is_empty:
                continue
            # Adjacent on the same line in either writing direction:
            # LTR chars step right, RTL chars step left.
            gap = (max(rect.x0 - merged[-1].x1, merged[-1].x0 - rect.x1)
                   if merged else 99.0)
            if merged and abs(merged[-1].y0 - rect.y0) < 2.0 and -1.0 <= gap < 12.0:
                merged[-1] |= rect
            else:
                merged.append(fitz.Rect(rect))
        return merged

    _LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st"}

    def joined_view(self) -> tuple[str, list[int]]:
        """Detector-friendly view: lines joined so wrapped tokens can match.

        Newlines become a single space (dropped after a hyphen), ligatures are
        expanded. Returns (text, mapping) where mapping[i] is the offset in
        self.text that joined char i came from; expanded ligature chars all
        map to the one original offset, whose rect covers the whole glyph.
        """
        out: list[str] = []
        mapping: list[int] = []
        text = self.text
        for i, ch in enumerate(text):
            if ch == "\n":
                prev = out[-1] if out else ""
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if not prev or not nxt or prev in " -" or prev == "\n":
                    continue
                if prev in "@._%+-/" or nxt in "@.":
                    continue  # token punctuation: jane.doe@\nexample.com joins tight
                out.append(" ")
                mapping.append(i)
                continue
            for expanded in self._LIGATURES.get(ch, ch):
                out.append(expanded)
                mapping.append(i)
        return "".join(out), mapping


@dataclass
class LineItem:
    """One extracted line: its bbox and (char, rect) pairs in stream order."""
    bbox: fitz.Rect
    chars: list[tuple[str, fitz.Rect | None]] = field(default_factory=list)
    index: int = 0  # content-stream position, used only as a row tie-break
    direction: tuple[float, float] = (1.0, 0.0)  # rawdict line "dir" vector
    wmode: int = 0  # rawdict writing mode (0=horizontal, 1=vertical)

    @property
    def text(self) -> str:
        return "".join(c for c, _ in self.chars)

    @property
    def is_vertical(self) -> bool:
        return self.wmode == 1 or abs(self.direction[1]) > abs(self.direction[0])

    @property
    def is_rtl(self) -> bool:
        return self.direction[0] < -0.5


def logical_chars(chars: list[tuple[str, "fitz.Rect | None"]],
                  direction: tuple[float, float],
                  ) -> list[tuple[str, "fitz.Rect | None"]]:
    """Return chars in logical (reading) order for a line.

    PyMuPDF reports chars in content-stream order, which for RTL or
    vertical text may be *visual* order (e.g. an RTL line stored
    left-to-right, or a vertical column stored bottom-to-top). When the
    glyph positions run opposite to the writing direction, reverse them
    so detectors see the text as read; when they already follow the
    direction, keep stream order untouched.
    """
    if len(chars) < 2:
        return chars
    dx, dy = direction
    # Judge monotonicity from glyphs that have a rect; rect-less chars
    # (rare) must not veto the reversal.
    if dx < -0.5:  # RTL: logical order runs right-to-left (decreasing x)
        centres = [(r.x0 + r.x1) / 2 for _, r in chars if r is not None]
        if len(centres) >= 2 and all(a <= b for a, b in zip(centres, centres[1:])):
            return list(reversed(chars))
    elif abs(dy) > abs(dx):  # vertical: logical order follows the dir vector
        centres = [(r.y0 + r.y1) / 2 for _, r in chars if r is not None]
        if len(centres) >= 2 and dy < 0 \
                and all(a <= b for a, b in zip(centres, centres[1:])):
            return list(reversed(chars))
        if len(centres) >= 2 and dy > 0 \
                and all(a >= b for a, b in zip(centres, centres[1:])):
            return list(reversed(chars))
    return chars


# Reading-order constants ported from RailReaderCore's XYCutPlusPlusResolver
# and TextLayoutAnalyzer (MIT, sjvrensburg/RailReaderCore), themselves
# validated on a 1,195-page corpus and from Liu, Li & Wei (2025),
# "XY-Cut++" (arXiv:2504.10258). Thresholds are page-relative where the
# original is; absolute point floors are theirs.
_MIN_COLUMN_GUTTER_PT = 7.0     # journals run gutters as tight as 8-11pt
_MIN_HORIZONTAL_GAP_PT = 4.0
_MIN_COLUMN_COVERAGE = 0.15     # each side's union height / region height
_MIN_COLUMN_WIDTH_FRAC = 0.15   # each side's width / region width
_PROJECTION_VALLEY_COVERAGE = 0.18
_PROJECTION_FLANK_COVERAGE = 0.45
_MARGIN_NOTE_WIDTH_FRAC = 0.5
_SPANNER_SIZE_FACTOR = 1.3
_DIVIDER_WIDTH_FRAC = 0.55
_BETWEEN_LINE_MULTIPLIER = 1.6  # Docstrum block join, x page line pitch
_BLOCK_OVERLAP_FRAC = 0.25


@dataclass
class BlockItem:
    """Lines grouped into a prose block (Docstrum-style, see build_blocks)."""
    bbox: fitz.Rect
    lines: list[LineItem] = field(default_factory=list)

    @property
    def stream_index(self) -> float:
        idx = sorted(li.index for li in self.lines)
        return idx[len(idx) // 2] if idx else 0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_s: float | None = None
    cur_e = 0.0
    for s, e in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s if cur_s is not None else 0.0)


def estimate_line_pitch(lines: list[LineItem]) -> float:
    """Page's own line pitch (Docstrum): median top-to-top distance to the
    nearest following line in the same column. Falls back to median height."""
    if len(lines) < 2:
        return max(1.0, lines[0].bbox.height) if lines else 1.0
    ordered = sorted(lines, key=lambda li: li.bbox.y0)
    pitches: list[float] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            d = b.bbox.y0 - a.bbox.y0
            if d < a.bbox.height * 0.5:
                continue  # same visual row / neighbouring column
            overlap = min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)
            if overlap < min(a.bbox.width, b.bbox.width) * _BLOCK_OVERLAP_FRAC:
                continue
            pitches.append(d)
            break
    if pitches:
        return max(1.0, _median(pitches))
    return max(1.0, _median([li.bbox.height for li in lines]))


def build_blocks(lines: list[LineItem]) -> list[BlockItem]:
    """Group lines into prose blocks: a line joins an open block when it
    follows within 1.6x the page line pitch and shares >=25% horizontal
    overlap with it. Blocks stay open so interleaved columns each collect
    their own lines (ported from TextLayoutAnalyzer.BuildBlocks)."""
    max_gap = estimate_line_pitch(lines) * _BETWEEN_LINE_MULTIPLIER
    blocks: list[BlockItem] = []
    for line in sorted(lines, key=lambda li: (li.bbox.y0, li.bbox.x0)):
        best, best_gap = None, float("inf")
        for block in blocks:
            gap = line.bbox.y0 - block.bbox.y1
            if gap > max_gap or line.bbox.y1 < block.bbox.y0:
                continue
            # A much wider/narrower line is a divider or a new column, not a
            # continuation: joining it would let a spanning title absorb both
            # columns into one block (union-bbox overlap then admits all).
            if max(block.bbox.width, line.bbox.width) > 2.0 * min(
                    block.bbox.width, line.bbox.width):
                continue
            overlap = (min(block.bbox.x1, line.bbox.x1)
                       - max(block.bbox.x0, line.bbox.x0))
            if overlap < min(block.bbox.width, line.bbox.width) * _BLOCK_OVERLAP_FRAC:
                continue
            if gap < best_gap:
                best, best_gap = block, gap
        if best is None:
            blocks.append(BlockItem(bbox=fitz.Rect(line.bbox), lines=[line]))
        else:
            best.lines.append(line)
            best.bbox |= line.bbox
    return blocks


def _region(blocks: list[BlockItem]) -> tuple[float, float, float, float]:
    return (min(b.bbox.y0 for b in blocks), max(b.bbox.y1 for b in blocks),
            min(b.bbox.x0 for b in blocks), max(b.bbox.x1 for b in blocks))


def _valid_column_sides(left: list[BlockItem], right: list[BlockItem],
                        reg_h: float, reg_w: float, *, min_each: int = 1) -> bool:
    if len(left) < min_each or len(right) < min_each:
        return False
    if reg_h > 0:
        for side in (left, right):
            covered = _union_length([(b.bbox.y0, b.bbox.y1) for b in side])
            if covered < _MIN_COLUMN_COVERAGE * reg_h:
                return False
    if reg_w > 0:
        for side in (left, right):
            w = max(b.bbox.x1 for b in side) - min(b.bbox.x0 for b in side)
            if w < _MIN_COLUMN_WIDTH_FRAC * reg_w:
                return False
    return True


def _find_column_split(blocks: list[BlockItem]) -> tuple[float, bool] | None:
    """Widest validated column gutter as (split_x, by_centre).

    First the clean sweep (no block straddles the gutter), then a
    projection-profile fallback that tolerates a thin/short crosser: the
    valley's height-coverage must be <=0.18 with flanks >=0.45.
    """
    if len(blocks) < 2:
        return None
    reg_top, reg_bottom, reg_left, reg_right = _region(blocks)
    reg_h, reg_w = reg_bottom - reg_top, reg_right - reg_left
    ordered = sorted(blocks, key=lambda b: b.bbox.x0)
    gaps: list[tuple[float, float]] = []
    max_end = ordered[0].bbox.x1
    for b in ordered[1:]:
        if b.bbox.x0 > max_end:
            gaps.append((max_end, b.bbox.x0))
        max_end = max(max_end, b.bbox.x1)
    for g0, g1 in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True):
        if g1 - g0 < _MIN_COLUMN_GUTTER_PT:
            break
        mid = (g0 + g1) / 2
        left = [b for b in blocks if b.bbox.x1 <= mid]
        right = [b for b in blocks if b.bbox.x0 >= mid]
        if _valid_column_sides(left, right, reg_h, reg_w):
            return mid, False
    # Projection-profile fallback (bins across the region width).
    if reg_h <= 0 or reg_w <= 0:
        return None
    bins = max(32, min(512, int(reg_w)))
    bin_w = reg_w / bins

    def coverage_at(x: float) -> float:
        iv = [(b.bbox.y0, b.bbox.y1) for b in blocks if b.bbox.x0 <= x <= b.bbox.x1]
        return _union_length(iv) / reg_h

    cov = [coverage_at(reg_left + (i + 0.5) * bin_w) for i in range(bins)]
    margin = max(1, int(bins * _MIN_COLUMN_WIDTH_FRAC))
    lo, hi = margin, bins - margin
    if hi - lo < 1:
        return None
    valley = min(range(lo, hi), key=lambda i: cov[i])
    if cov[valley] > _PROJECTION_VALLEY_COVERAGE:
        return None
    if max(cov[lo:valley], default=0) < _PROJECTION_FLANK_COVERAGE:
        return None
    if max(cov[valley + 1:hi], default=0) < _PROJECTION_FLANK_COVERAGE:
        return None
    split_x = reg_left + (valley + 0.5) * bin_w
    left = [b for b in blocks if (b.bbox.x0 + b.bbox.x1) / 2 < split_x]
    right = [b for b in blocks if (b.bbox.x0 + b.bbox.x1) / 2 >= split_x]
    if _valid_column_sides(left, right, reg_h, reg_w, min_each=2):
        return split_x, True
    return None


def _find_horizontal_split(blocks: list[BlockItem]) -> float | None:
    ordered = sorted(blocks, key=lambda b: b.bbox.y0)
    max_end, best = ordered[0].bbox.y1, None
    for b in ordered[1:]:
        gap = b.bbox.y0 - max_end
        if gap >= _MIN_HORIZONTAL_GAP_PT and (best is None or gap > best[0]):
            best = (gap, (max_end + b.bbox.y0) / 2)
        max_end = max(max_end, b.bbox.y1)
    return best[1] if best else None


def _is_divider(block: BlockItem, blocks: list[BlockItem], reg_w: float) -> bool:
    """Full-width block in its own band, with narrower blocks present."""
    if reg_w <= 0 or block.bbox.width < _DIVIDER_WIDTH_FRAC * reg_w:
        return False
    if not any(b is not block and b.bbox.width < _DIVIDER_WIDTH_FRAC * reg_w
               for b in blocks):
        return False
    return not any(b is not block and b.bbox.y0 < block.bbox.y1
                   and block.bbox.y0 < b.bbox.y1 for b in blocks)


def _order_leaf(blocks: list[BlockItem]) -> list[BlockItem]:
    """Rows by vertical overlap; within a row, content-stream tie-break."""
    rows: list[list[BlockItem]] = []
    row_top = row_bottom = 0.0
    for b in sorted(blocks, key=lambda b: b.bbox.y0):
        if rows:
            overlap = min(row_bottom, b.bbox.y1) - max(row_top, b.bbox.y0)
            min_h = min(b.bbox.height, row_bottom - row_top)
            if overlap > 0.5 * min_h:
                rows[-1].append(b)
                row_top, row_bottom = min(row_top, b.bbox.y0), max(row_bottom, b.bbox.y1)
                continue
        rows.append([b])
        row_top, row_bottom = b.bbox.y0, b.bbox.y1
    ordered: list[BlockItem] = []
    for row in rows:
        ordered += sorted(row, key=lambda b: (b.stream_index, b.bbox.x0))
    return ordered


def _cut(blocks: list[BlockItem]) -> list[BlockItem]:
    if len(blocks) <= 1:
        return list(blocks)
    _, _, reg_left, reg_right = _region(blocks)
    reg_w = reg_right - reg_left
    # Full-width dividers (title, mid heading, footnote band) split the
    # region into horizontal bands before any column cut is attempted.
    dividers = [b for b in blocks if _is_divider(b, blocks, reg_w)]
    if dividers:
        divider = min(dividers, key=lambda b: b.bbox.y0)
        rest = [b for b in blocks if b is not divider]
        above = [b for b in rest if b.bbox.y1 <= divider.bbox.y0]
        below = [b for b in rest if b.bbox.y1 > divider.bbox.y0]
        return _cut(above) + [divider] + _cut(below)
    split = _find_column_split(blocks)
    if split is not None:
        split_x, by_centre = split
        if by_centre:
            left = [b for b in blocks if (b.bbox.x0 + b.bbox.x1) / 2 < split_x]
            right = [b for b in blocks if (b.bbox.x0 + b.bbox.x1) / 2 >= split_x]
        else:
            left = [b for b in blocks if b.bbox.x1 <= split_x]
            right = [b for b in blocks if b.bbox.x1 > split_x]
        if left and right:
            return _cut(left) + _cut(right)
    split_y = _find_horizontal_split(blocks)
    if split_y is not None:
        top = [b for b in blocks if b.bbox.y1 <= split_y]
        bottom = [b for b in blocks if b.bbox.y1 > split_y]
        if top and bottom:
            return _cut(top) + _cut(bottom)
    return _order_leaf(blocks)


def _pre_mask(blocks: list[BlockItem]) -> tuple[list[BlockItem], list[BlockItem]]:
    """Lift margin notes and clipping spanners out before cutting."""
    if len(blocks) < 3:
        return [], list(blocks)
    med_w = _median([b.bbox.width for b in blocks])
    med_h = _median([b.bbox.height for b in blocks])
    masked: list[BlockItem] = []
    body: list[BlockItem] = []
    for b in blocks:
        narrow = b.bbox.width <= med_w * _MARGIN_NOTE_WIDTH_FRAC
        margin_note = narrow and any(
            c is not b and c.bbox.width > b.bbox.width * 1.5
            and c.bbox.y0 < b.bbox.y1 and b.bbox.y0 < c.bbox.y1
            and (c.bbox.x1 <= b.bbox.x0 or b.bbox.x1 <= c.bbox.x0)
            for c in blocks)
        large = (b.bbox.width >= med_w * _SPANNER_SIZE_FACTOR
                 or b.bbox.height >= med_h * _SPANNER_SIZE_FACTOR)
        clipping = large and any(
            c is not b
            and min(b.bbox.x1, c.bbox.x1) - max(b.bbox.x0, c.bbox.x0) > 0
            and min(b.bbox.y1, c.bbox.y1) - max(b.bbox.y0, c.bbox.y0) > 0
            for c in blocks)
        (masked if (margin_note or clipping) else body).append(b)
    if not body:
        return [], list(blocks)
    return masked, body


def _reinsert(ordered: list[BlockItem], masked: list[BlockItem]) -> list[BlockItem]:
    """Re-insert masked blocks after their nearest ordered neighbour."""
    ordered = list(ordered)
    for m in sorted(masked, key=lambda b: (b.bbox.y0, b.bbox.x0)):
        if not ordered:
            ordered.append(m)
            continue
        cx, cy = (m.bbox.x0 + m.bbox.x1) / 2, (m.bbox.y0 + m.bbox.y1) / 2
        anchor = min(range(len(ordered)),
                     key=lambda i: ((ordered[i].bbox.x0 + ordered[i].bbox.x1) / 2 - cx) ** 2
                     + ((ordered[i].bbox.y0 + ordered[i].bbox.y1) / 2 - cy) ** 2)
        ordered.insert(anchor + 1, m)
    return ordered


def order_blocks(blocks: list[BlockItem]) -> list[BlockItem]:
    masked, body = _pre_mask(blocks)
    return _reinsert(_cut(body), masked)


def xy_cut_order(items: list[LineItem]) -> list[LineItem]:
    """Reading order: Docstrum block grouping, then XY-Cut++ on blocks.

    Ported from RailReaderCore (MIT): TextLayoutAnalyzer's page-derived
    grouping plus XYCutPlusPlusResolver's validated/projection gutters,
    divider handling and margin-note masking. Expands ordered blocks back
    to lines (top-to-bottom within a block).
    """
    lines = [i for i in items if i.chars]
    if len(lines) <= 1:
        return lines
    if all(li.is_vertical for li in lines):
        # Vertical (e.g. CJK) columns: read top-to-bottom within a column,
        # columns right-to-left.
        return sorted(lines, key=lambda li: (-li.bbox.x0, li.bbox.y0))
    if any(li.is_rtl for li in lines):
        # Mirror x and reuse the LTR machinery: RTL columns then read
        # right-to-left with rows top-to-bottom.
        mirrored = [LineItem(
            bbox=fitz.Rect(-li.bbox.x1, li.bbox.y0, -li.bbox.x0, li.bbox.y1),
            chars=li.chars, index=li.index, direction=li.direction,
            wmode=li.wmode) for li in lines]
        by_id = {id(m): o for m, o in zip(mirrored, lines)}
        return [by_id[id(m)] for block in order_blocks(build_blocks(mirrored))
                for m in sorted(block.lines,
                                key=lambda li: (li.bbox.y0, li.bbox.x0))]
    ordered: list[LineItem] = []
    for block in order_blocks(build_blocks(lines)):
        ordered += sorted(block.lines, key=_within_block_key)
    return ordered


def _within_block_key(li: LineItem) -> tuple[float, float]:
    if li.is_vertical:
        return (-li.bbox.x0, li.bbox.y0)
    return (li.bbox.y0, li.bbox.x0)


def extract_page(page: fitz.Page, page_index: int) -> PageText:
    out = PageText(page=page_index)
    out.has_images = bool(page.get_images(full=True))
    # Accumulate in a list and join once: per-char `text += ch` is O(n^2)
    # string copying on large pages.
    chars: list[str] = []

    def emit_char(ch: str, rect: fitz.Rect | None) -> None:
        chars.append(ch)
        out.rects.append(rect)
        if rect is not None:
            out.char_count += 1

    lines: list[LineItem] = []
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            direction = tuple(float(v) for v in line.get("dir", (1.0, 0.0)))
            item = LineItem(bbox=fitz.Rect(line["bbox"]), index=len(lines),
                            direction=(direction[0], direction[1]),
                            wmode=int(line.get("wmode", 0)))
            raw_chars: list[tuple[str, fitz.Rect | None]] = []
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    c = char.get("c", "")
                    if not c:
                        continue
                    bbox = char.get("bbox")
                    raw_chars.append((c, fitz.Rect(bbox) if bbox else None))
            item.chars = logical_chars(raw_chars, item.direction)
            if item.chars:
                lines.append(item)
    for li, item in enumerate(xy_cut_order(lines)):
        if li:
            emit_char("\n", None)
        for ch, rect in item.chars:
            emit_char(ch, rect)
    out.text = "".join(chars)
    return out


def is_probably_scanned(pt: PageText) -> bool:
    """No text layer but page content exists -> scanned/image-only page."""
    return pt.char_count == 0 and pt.has_images
