"""Page-range parsing. Specs are 1-based and inclusive (see report §4)."""
from __future__ import annotations

from collections.abc import Iterable

from . import PdfCloakError


def parse_page_spec(spec: str | None, page_count: int) -> list[int]:
    """Return sorted 0-based page indices for `spec` in a `page_count`-page doc."""
    if page_count <= 0:
        raise PdfCloakError("document has no pages")
    text = (spec or "all").strip().lower()
    if text in ("all", ""):
        return list(range(page_count))

    def endpoint(tok: str, *, is_end: bool) -> int:
        tok = tok.strip()
        if tok in ("last", "end"):
            return page_count
        if tok == "first":
            return 1
        if tok == "":
            return page_count if is_end else 1
        try:
            n = int(tok)
        except ValueError:
            raise PdfCloakError(f"invalid page token: {tok!r} in --pages {spec!r}") from None
        if n == 0 or n < 0:
            raise PdfCloakError(f"pages are 1-based; got {n} in --pages {spec!r}")
        return n

    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise PdfCloakError(f"empty component in --pages {spec!r}")
        if part in ("all",):
            return list(range(page_count))
        if part == "-":
            raise PdfCloakError(
                f"bare '-' is not a page spec in --pages {spec!r} (use 'all' or '-N')")
        candidates: Iterable[int]
        if "-" in part[1:] or (part.startswith("-") and len(part) > 1) or part.endswith("-"):
            left, _, right = part.partition("-")
            lo = endpoint(left, is_end=False) if left.strip() else 1
            hi = endpoint(right, is_end=True) if right.strip() else page_count
            if not left.strip():
                hi = min(hi, page_count)  # '-99' clamps like '3-' does
            if lo > hi:
                raise PdfCloakError(f"reversed page range {part!r} in --pages {spec!r}")
            candidates = range(lo, hi + 1)
        else:
            candidates = [endpoint(part, is_end=False)]
        for n in candidates:
            if n < 1 or n > page_count:
                raise PdfCloakError(
                    f"page {n} out of range (document has {page_count} pages)")
            pages.add(n - 1)
    if not pages:
        raise PdfCloakError(f"no pages selected by --pages {spec!r}")
    return sorted(pages)
