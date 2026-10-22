"""Shared span type returned by every detector."""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field


def masked_text(text: str) -> str:
    """Redacted glimpse of a match, safe to show without --show-matches.

    Keeps only enough shape to recognise the hit (``j***@e***.com``,
    ``+***4``) so a reviewer can triage scan -> redact without printing PII.
    """
    if not text:
        return ""
    if "@" in text and "." in text.rsplit("@", 1)[-1]:
        local, _, domain = text.partition("@")
        host, _, tld = domain.rpartition(".")
        # Short local/host parts would be fully reconstructable from the
        # first-letter glimpse, so mask them entirely.
        if len(local) <= 2 or len(host) <= 2:
            return "***"
        return f"{local[:1]}***@{host[:1]}***.{tld}"
    if len(text) <= 8:
        return "***"
    return f"{text[:1]}***{text[-1:]}"


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

    @property
    def is_rule(self) -> bool:
        return self.source == "regex" or self.score >= 2.0

    def digest(self) -> str:
        """Short hash for audit reports (never log raw PII by default).

        An unsalted SHA-256 of low-entropy PII (emails, phone numbers) is
        guess-confirmable offline, so when PDFCLOAK_REPORT_HMAC_KEY is set
        the digest is keyed HMAC instead.
        """
        key = os.environ.get("PDFCLOAK_REPORT_HMAC_KEY")
        data = self.text.encode("utf-8", "replace")
        if key:
            return hmac.new(key.encode(), data, hashlib.sha256).hexdigest()[:12]
        return hashlib.sha256(data).hexdigest()[:12]

    def report_entry(self, *, show_text: bool = False) -> dict:
        entry: dict = {
            "page": self.page + 1 if self.page >= 0 else None,
            "label": self.label,
            # Regex rules carry a sentinel score of 2.0 (priority, not
            # confidence); reporting it beside 0-1 model scores reads as
            # >100% confidence, so rules report score:null + priority.
            "score": None if self.is_rule else round(self.score, 4),
            "priority": "rule" if self.is_rule else "model",
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "masked_text": masked_text(self.text),
        }
        # Only emit a digest when it is keyed: an unsalted hash of an
        # email/phone number can be confirmed by guessing offline.
        if os.environ.get("PDFCLOAK_REPORT_HMAC_KEY"):
            entry["digest"] = self.digest()
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
