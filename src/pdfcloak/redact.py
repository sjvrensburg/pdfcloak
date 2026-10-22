"""Scan / redact / verify pipeline on top of PyMuPDF's true redaction."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pymupdf as fitz

from . import PdfCloakError
from .detectors import Detector
from .extract import PageText, extract_page
from .models import Span, masked_text, resolve_overlaps

REDACT_PAD = 0.5  # points; PyMuPDF deletes any char *overlapping* the rect


def detect_on_pages(doc: fitz.Document, page_indices: list[int],
                    detectors: list[Detector], *, labels: list[str],
                    threshold: float,
                    progress: Callable[[int, int, int], None] | None = None,
                    ) -> tuple[list[Span], list[PageText], list[int]]:
    """Run detectors per selected page. Returns (spans, page_texts, unchecked).

    Unchecked pages are those with no extractable text at all (scanned,
    image-only, blank, or text rendered as vector paths): they were not
    inspected, so callers must not report them clean.
    """
    all_spans: list[Span] = []
    pages: list[PageText] = []
    scanned: list[int] = []
    total = len(page_indices)
    # Cache detector results per (detector, text) within this pass: the raw
    # and line-joined views are often identical, and re-asking a model
    # detector (an HTTP round-trip) for the same text doubles AI latency.
    cache: dict[tuple[int, int, str], list[Span]] = {}

    def run(det: Detector, text: str, idx: int) -> list[Span]:
        key = (id(det), idx, text)
        if key not in cache:
            cache[key] = det.detect(text, page=idx, labels=labels, threshold=threshold)
        return cache[key]

    for n, idx in enumerate(page_indices, 1):
        if progress is not None:
            progress(idx, n, total)
        pt = extract_page(doc[idx], idx)
        pages.append(pt)
        if pt.char_count == 0:
            scanned.append(idx)
            continue  # no text layer to detect on (OCR is a future phase)
        joined, mapping = pt.joined_view()
        for det in detectors:
            all_spans.extend(run(det, pt.text, idx))
            # Also detect on the line-joined view so tokens wrapped across
            # lines (jane.doe@\nexample.com) match; map offsets back.
            if joined != pt.text:
                for js in run(det, joined, idx):
                    if 0 <= js.start < len(mapping) and js.end - 1 < len(mapping):
                        all_spans.append(Span(
                            idx, mapping[js.start], mapping[js.end - 1] + 1,
                            js.label, js.score, js.source, js.text))
    return resolve_overlaps(all_spans), pages, scanned


def span_text(pt: PageText, span: Span) -> str:
    return pt.text[span.start:span.end]


def apply_redactions(doc: fitz.Document, spans: list[Span],
                     page_texts: dict[int, PageText], *, replacement: str) -> int:
    """Queue one redaction annot per line-fragment, then apply per page."""
    applied_pages: set[int] = set()
    by_page: dict[int, list[Span]] = {}
    for span in spans:
        by_page.setdefault(span.page, []).append(span)
    for page_idx, page_spans in by_page.items():
        page = doc[page_idx]
        pt = page_texts[page_idx]
        for span in page_spans:
            text = replacement_label(span) if replacement == "label" else None
            for rect in pt.rects_for_span(span.start, span.end):
                padded = fitz.Rect(rect.x0 - REDACT_PAD, rect.y0 - REDACT_PAD,
                                   rect.x1 + REDACT_PAD, rect.y1 + REDACT_PAD)
                # Fit the replacement label to the box it replaces: a fixed
                # 11pt label overflows narrow columns and tiny spans.
                fontsize = max(4.0, min(11.0, rect.height * 0.7)) if text else 11
                page.add_redact_annot(padded, text=text, fontsize=fontsize,
                                      text_color=(1, 1, 1), fill=(0, 0, 0),
                                      cross_out=False)
                applied_pages.add(page_idx)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_PIXELS,  # type: ignore[attr-defined]
            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,  # type: ignore[attr-defined]
            text=fitz.PDF_REDACT_TEXT_REMOVE)  # type: ignore[attr-defined]
    return len(applied_pages)


def replacement_label(span: Span) -> str:
    return "[" + span.label.strip().upper().replace(" ", "_") + "]"


def count_carriers(doc: fitz.Document, *, keep_metadata: bool = False) -> dict:
    """Count hidden PII carriers scrub_document would remove (dry-run view)."""
    counts = {
        "metadata_cleared": not keep_metadata,
        "toc_entries": 0 if keep_metadata else len(doc.get_toc()),
        "attachments": doc.embfile_count(),
        "annotations": 0,
        "widgets": 0,
        "links": 0,
        "document_actions_cleared": not keep_metadata,
    }
    for page in [doc[i] for i in range(doc.page_count)]:
        counts["annotations"] += len(list(page.annots() or []))
        counts["widgets"] += len(list(page.widgets() or []))
        counts["links"] += len(page.get_links())
    return counts


_LANG_RE = __import__("re").compile(r"^[A-Za-z][A-Za-z-]{0,34}$")


def scrub_document(doc: fitz.Document, redacted_pages: list[int], *,
                   keep_metadata: bool = False, lang: str = "en") -> dict:
    """Remove PII hiding outside page text (report §4 step 7).

    Annotations, form widgets and link targets are scrubbed on *all* pages,
    not just redacted ones: they are invisible carriers (a text annot, a
    mailto: URI, a bookmark title) that would otherwise survive a page-scoped
    redaction while metadata/attachments are already removed document-wide.
    Widgets are deleted outright — clearing field_value alone leaves the
    rendered appearance stream in place. Visible page *text* on unselected
    pages is still left alone; that is what --pages means.

    Returns an audit dict of what was removed, so reports and --dry-run can
    name this destructive side effect instead of surprising the user.
    Document-level JavaScript / open actions are cleared with the metadata;
    a neutral title and the document language are (re)set so screen readers
    keep a title/language after scrubbing.
    """
    if not _LANG_RE.match(lang or ""):
        raise PdfCloakError(f"invalid document language {lang!r}")
    counts = count_carriers(doc, keep_metadata=keep_metadata)
    if not keep_metadata:
        doc.set_metadata({k: "" for k in
                          ("title", "author", "subject", "keywords", "creator",
                           "producer", "trapped")})
        doc.set_metadata({"title": "Redacted document"})
        try:
            doc.del_xml_metadata()
        except Exception:
            pass
        try:
            doc.set_toc([])
        except Exception:
            pass
        # Clear document-level actions/scripts that can carry data or code.
        try:
            catalog = doc.pdf_catalog()
            for key in ("Names", "OpenAction", "AA"):
                doc.xref_set_key(catalog, key, "null")
            if lang:
                doc.xref_set_key(catalog, "Lang", f"({lang})")
        except Exception:
            pass
    for i in range(doc.embfile_count() - 1, -1, -1):
        doc.embfile_del(i)
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        for annot in list(page.annots() or []):
            page.delete_annot(annot)
        for widget in list(page.widgets() or []):
            page.delete_widget(widget)
        for link in list(page.get_links()):
            page.delete_link(link)
    return counts


def make_preview(doc: fitz.Document, spans: list[Span],
                 page_texts: dict[int, PageText], out_path: str | Path) -> None:
    """Write an annotated preview PDF: highlights show what *would* be removed.

    Call before apply_redactions on the same document; the highlight annots
    are removed again by scrub_document before the real output is saved.
    Annot contents carry only the label and masked text, never raw PII.
    """
    by_page: dict[int, list[Span]] = {}
    for span in spans:
        by_page.setdefault(span.page, []).append(span)
    for page_idx, page_spans in by_page.items():
        page = doc[page_idx]
        pt = page_texts[page_idx]
        for span in page_spans:
            for rect in pt.rects_for_span(span.start, span.end):
                annot = page.add_highlight_annot(rect)
                annot.set_info(title="pdfcloak",
                               content=f"{span.label}: {masked_text(span.text)}")
                annot.update()
    save_safely(doc, out_path)


def save_safely(doc: fitz.Document, out_path: str | Path) -> None:
    """Save atomically with owner-only permissions.

    Never incremental: original bytes must not survive in the output. The
    file is written to a temp file in the destination directory, fsynced,
    then renamed over the target, so a crash or a failed verification never
    leaves a partial file at the shareable path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        doc.save(tmp, garbage=4, deflate=True, clean=True)
        os.chmod(tmp, 0o600)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        try:  # make the rename itself crash-durable
            dir_fd = os.open(out.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def verify_output(out_path: str | Path, page_indices: list[int],
                  detectors: list[Detector], *, labels: list[str],
                  threshold: float,
                  original_texts: list[str]) -> tuple[list[Span], list[Span]]:
    """Re-open the output and check that the originals are really gone.

    Returns (survivors, new_detections). A survivor is a detection whose text
    equals an originally-redacted string, or that string appearing in the raw
    bytes. New detections are fresh detector hits on the mangled text (e.g. the
    model flagging "at" in "Contact [bar] at [bar]") — reported, but they are
    not proof the redaction failed; run `pdfcloak verify` to chase them.
    """
    survivors: list[Span] = []
    new_detections: list[Span] = []
    originals = {t for t in original_texts if t}

    def is_survivor(text: str) -> bool:
        # Exact match first (set lookup), then partial leftover: a fragment
        # of an original (e.g. "ane@example.com") or an original inside a
        # larger detection.
        if not text:
            return False
        if text in originals:
            return True
        return any(text in o or o in text for o in originals)

    with fitz.open(out_path) as doc:
        spans, page_texts, _ = detect_on_pages(doc, page_indices, detectors,
                                               labels=labels, threshold=threshold)
        by_page = {pt.page: pt for pt in page_texts}
        seen_survivor_texts: set[str] = set()
        for span in spans:
            text = span.text or (by_page[span.page].text[span.start:span.end]
                                 if span.page in by_page else "")
            filled = Span(span.page, span.start, span.end, span.label,
                          span.score, span.source, text)
            if is_survivor(text):
                survivors.append(filled)
                seen_survivor_texts.add(text)
            else:
                new_detections.append(filled)
        # Raw-bytes check is a bonus only: save_safely deflates streams, so
        # absence here proves nothing. Report page=None (Span page=-1).
        # Streamed in chunks (with overlap) so large outputs are not loaded
        # whole into memory.
        needles = {t.encode("utf-8", "ignore"): t
                   for t in originals - seen_survivor_texts if t}
        if needles:
            max_len = max(len(n) for n in needles)
            tail = b""
            found: set[str] = set()
            with open(out_path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    blob = tail + chunk
                    for needle, text in needles.items():
                        if needle in blob:
                            found.add(text)
                    tail = blob[-(max_len - 1):] if max_len > 1 else b""
            for text in sorted(found):
                survivors.append(Span(-1, 0, len(text), "raw-bytes", 1.0,
                                      "verify", text))
    return survivors, new_detections
