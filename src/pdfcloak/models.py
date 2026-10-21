"""Shared span type returned by every detector."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    page: int          # 0-based page index
    start: int         # char offset into the page string, inclusive
    end: int           # char offset, exclusive
    label: str
    score: float
    source: str        # detector name, e.g. "regex", "clipcloak"
    text: str = field(default="", compare=False)

    @property
    def length(self) -> int:
        return self.end - self.start

    def digest(self) -> str:
        """Short hash for audit reports (never log raw PII by default)."""
        return hashlib.sha256(self.text.encode("utf-8", "replace")).hexdigest()[:12]

    def report_entry(self, *, show_text: bool = False) -> dict:
        entry: dict = {
            "page": self.page + 1 if self.page >= 0 else None,
            "label": self.label,
            "score": round(self.score, 4),
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "sha256": self.digest(),
        }
        if show_text:
            entry["text"] = self.text
        return entry


def _residual(span: Span, start: int, end: int) -> Span:
    text = span.text[start - span.start:end - span.start] if len(span.text) == span.length else span.text
    return Span(span.page, start, end, span.label, span.score, span.source, text)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Pool spans per page: highest score, then longest, then earliest wins.

    Regex rules use score 2.0 (above any model score), so rules always win.
    A losing span is not dropped whole: its non-overlapping head/tail survive
    as residual spans (same policy as ClipCloak's `resolve_entities`), so a
    partially-overlapped detection cannot leave a PII fragment behind.
    """
    ordered = sorted(spans, key=lambda s: (s.page, -s.score, -(s.end - s.start), s.start))
    kept: list[Span] = []
    taken: dict[int, list[tuple[int, int]]] = {}
    for span in ordered:
        if span.end <= span.start:
            continue
        pieces = [(span.start, span.end)]
        for s, e in taken.get(span.page, []):
            remaining = []
            for a, b in pieces:
                if e <= a or s >= b:
                    remaining.append((a, b))
                else:
                    if a < s:
                        remaining.append((a, s))
                    if e < b:
                        remaining.append((e, b))
            pieces = remaining
            if not pieces:
                break
        intervals = taken.setdefault(span.page, [])
        for a, b in pieces:
            kept.append(_residual(span, a, b))
            intervals.append((a, b))
    return sorted(kept, key=lambda s: (s.page, s.start))
