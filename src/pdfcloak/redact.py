"""Scan / redact / verify pipeline on top of PyMuPDF's true redaction."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from . import PdfCloakError
from .detectors import Detector
from .extract import PageText, extract_page, is_probably_scanned
from .models import Span, resolve_overlaps

REDACT_PAD = 0.5  # points; PyMuPDF deletes any char *overlapping* the rect


def detect_on_pages(doc: fitz.Document, page_indices: list[int],
                    detectors: list[Detector], *, labels: list[str],
                    threshold: float) -> tuple[list[Span], list[PageText], list[int]]:
    """Run detectors per selected page. Returns (spans, page_texts, scanned_pages)."""
    all_spans: list[Span] = []
    pages: list[PageText] = []
    scanned: list[int] = []
    for idx in page_indices:
        pt = extract_page(doc[idx], idx)
        pages.append(pt)
        if is_probably_scanned(pt):
            scanned.append(idx)
            continue  # no text layer to detect on (OCR is a future phase)
        for det in detectors:
            all_spans.extend(det.detect(pt.text, page=idx, labels=labels, threshold=threshold))
            # Also detect on the line-joined view so tokens wrapped across
            # lines (jane.doe@\nexample.com) match; map offsets back.
            joined, mapping = pt.joined_view()
            if joined != pt.text:
                for js in det.detect(joined, page=idx, labels=labels, threshold=threshold):
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
                page.add_redact_annot(padded, text=text, fill=(0, 0, 0),
                                      cross_out=False)
                applied_pages.add(page_idx)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_PIXELS,
            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
            text=fitz.PDF_REDACT_TEXT_REMOVE)
    return len(applied_pages)


def replacement_label(span: Span) -> str:
    return "[" + span.label.strip().upper().replace(" ", "_") + "]"


def scrub_document(doc: fitz.Document, redacted_pages: list[int], *,
                   keep_metadata: bool = False) -> None:
    """Remove PII hiding outside page text (report §4 step 7).

    Annotations, form widgets and link targets are scrubbed on *all* pages,
    not just redacted ones: they are invisible carriers (a text annot, a
    mailto: URI, a bookmark title) that would otherwise survive a page-scoped
    redaction while metadata/attachments are already removed document-wide.
    Widgets are deleted outright — clearing field_value alone leaves the
    rendered appearance stream in place. Visible page *text* on unselected
    pages is still left alone; that is what --pages means.
    """
    if not keep_metadata:
        doc.set_metadata({k: "" for k in
                          ("title", "author", "subject", "keywords", "creator",
                           "producer", "trapped")})
        try:
            doc.del_xml_metadata()
        except Exception:
            pass
        try:
            doc.set_toc([])
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


def save_safely(doc: fitz.Document, out_path: str | Path) -> None:
    """Never incremental: original bytes must not survive in the output."""
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)


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
        # Exact match, or a partial leftover: a fragment of an original
        # (e.g. "ane@example.com") or an original inside a larger detection.
        return bool(text) and any(text in o or o in text for o in originals)

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
        raw = Path(out_path).read_bytes()
        for text in originals - seen_survivor_texts:
            if text.encode("utf-8", "ignore") in raw:
                survivors.append(Span(-1, 0, len(text), "raw-bytes", 1.0,
                                      "verify", text))
    return survivors, new_detections
