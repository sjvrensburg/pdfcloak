"""Detectors: regex, ClipCloak sidecar (GLiNER2), generic single-graph ONNX.

Every detector takes plain text and returns character-offset Spans.
None of them knows anything about PDFs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from . import PdfCloakError
from .models import Span
from .rules import MAX_MATCHES_PER_RULE, RULE_SCORE, Rule, compile_rules

DEFAULT_LABELS = ["name", "street address", "email", "phone_num", "id_num", "url", "username"]


class Detector(Protocol):
    name: str

    def detect(self, text: str, *, page: int, labels: list[str], threshold: float) -> list[Span]:
        ...


class RegexDetector:
    name = "regex"

    def __init__(self, rules: list[Rule]):
        self.compiled = compile_rules(rules)

    def detect(self, text: str, *, page: int, labels: list[str], threshold: float) -> list[Span]:
        spans: list[Span] = []
        for rule, pattern in self.compiled:
            count = 0
            try:
                matches = pattern.finditer(text, timeout=2.0)
                for m in matches:
                    if m.start() == m.end():
                        continue
                    spans.append(Span(page, m.start(), m.end(), rule.name,
                                      RULE_SCORE, self.name, m.group()))
                    count += 1
                    if count >= MAX_MATCHES_PER_RULE:
                        # Fail closed: silently truncating would let the
                        # excess matches survive and still report clean.
                        raise PdfCloakError(
                            f"rule {rule.name!r} hit the {MAX_MATCHES_PER_RULE}-match "
                            f"cap on page {page + 1}; narrow the pattern or the "
                            "page range — refusing to redact partially")
            except TimeoutError:
                raise PdfCloakError(
                    f"rule {rule.name!r} timed out on page {page + 1}; simplify the pattern"
                ) from None
        return spans


def _reconcile(text: str, start: int, end: int, expected: str) -> tuple[int, int] | None:
    """ClipCloak mixes char offsets (model) and byte offsets (its regex rules).

    Validate the slice; if it does not match, try char offsets as byte offsets,
    then fall back to locating the expected text near the given position.
    """
    if 0 <= start < end <= len(text) and text[start:end] == expected:
        return start, end
    try:
        raw = text.encode("utf-8")
        decoded = raw[start:end].decode("utf-8")
        if decoded == expected:
            return len(raw[:start].decode("utf-8")), len(raw[:end].decode("utf-8"))
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Nearest occurrence to the supplied offset (never a blind global first
    # hit, which misattributes duplicates); drop the entity if none is near.
    candidates: list[int] = []
    pos = text.find(expected)
    while pos >= 0:
        candidates.append(pos)
        pos = text.find(expected, pos + 1)
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: abs(c - start))
    if abs(nearest - start) > max(64, len(expected) * 4):
        return None
    return nearest, nearest + len(expected)


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _validate_sidecar_url(base_url: str, token: str | None) -> str:
    """Refuse to spray page text + Bearer [REDACTED] at an arbitrary URL.

    Loopback http is fine (that is the local sidecar). Anything else needs
    https, and a token over plain http to a remote host is always refused.
    Set PDFCLOAK_ALLOW_REMOTE_SIDECAR=1 to explicitly trust a remote https
    endpoint.
    """
    import os
    import urllib.parse

    url = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PdfCloakError(f"--sidecar-url must be an http(s) URL, got {base_url!r}")
    host = parsed.hostname.lower()
    if host in _LOOPBACK_HOSTS:
        return url
    if parsed.scheme != "https":
        raise PdfCloakError(
            f"--sidecar-url {base_url!r} is not loopback; refusing to send "
            "page text (and any Bearer [REDACTED]) over plain http — use https")
    if os.environ.get("PDFCLOAK_ALLOW_REMOTE_SIDECAR") != "1":
        raise PdfCloakError(
            f"--sidecar-url {base_url!r} is a remote endpoint; page text would "
            "leave this machine. Set PDFCLOAK_ALLOW_REMOTE_SIDECAR=1 to trust it")
    return url


_CHUNK_OVERLAP = 512  # chars; an entity straddling a cut must appear whole
# in at least one chunk (entities longer than this are pathological).


def _text_chunks(text: str, max_bytes: int = 35_000) -> list[tuple[int, str]]:
    """Split text into overlapping (char_offset, chunk) under the 40k cap."""
    if len(text.encode("utf-8", "ignore")) <= max_bytes:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    start, size = 0, 0
    for i, ch in enumerate(text):
        size += len(ch.encode("utf-8", "ignore"))
        if size > max_bytes and i > start:
            chunks.append((start, text[start:i]))
            start = max(start, i - _CHUNK_OVERLAP)
            size = len(text[start:i].encode("utf-8", "ignore"))
    chunks.append((start, text[start:]))
    return chunks


class ClipCloakDetector:
    """HTTP client for the vendored ClipCloak daemon (Option 2 sidecar)."""

    name = "clipcloak"

    def __init__(self, base_url: str, token: str | None = None, *,
                 timeout: float = 600.0, retries: int = 2):
        self.base_url = _validate_sidecar_url(base_url, token)
        self.token = token
        self.timeout = timeout
        self.retries = retries

    def health(self, timeout: float = 5.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as r:
                if r.status != 200:
                    return False
                return json.loads(r.read().decode("utf-8")).get("ok") is True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def identity(self, timeout: float = 5.0) -> bool:
        """Authenticated identity probe: proves the listener is our sidecar.

        Plain /health is identity-free, so anything that grabbed the port
        could answer it. /identity requires the bearer token and names the
        service. Fail closed: a daemon without /identity (404) is not
        trusted — a /health fallback lets a port-squatter impersonate it.
        """
        req = urllib.request.Request(
            f"{self.base_url}/identity",
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    return False
                return json.loads(r.read().decode("utf-8")).get("service") == "clipcloak"
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
            return False

    def _classify(self, text: str, *, page: int, threshold: float) -> dict:
        """One /classify POST with bounded retries on transient failures."""
        import time
        body = json.dumps({"text": text, "threshold": threshold}).encode()
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/classify", data=body,
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {self.token}"} if self.token else {})},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise PdfCloakError("ClipCloak sidecar returned a non-object response")
                return payload
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                if exc.code < 500 or attempt >= self.retries:
                    raise PdfCloakError(
                        f"ClipCloak sidecar HTTP {exc.code} on page {page + 1}"
                        + (f": {detail}" if detail else "")) from None
                last_exc = exc
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break
            except ValueError as exc:
                raise PdfCloakError(
                    f"ClipCloak sidecar returned invalid JSON on page {page + 1}") from exc
            time.sleep(0.2 * (attempt + 1))
        raise PdfCloakError(
            f"ClipCloak sidecar unreachable ({last_exc}); is it running?") from None

    def detect(self, text: str, *, page: int, labels: list[str], threshold: float) -> list[Span]:
        wanted = set(labels)
        spans: list[Span] = []
        dropped = 0
        for offset, chunk in _text_chunks(text):
            payload = self._classify(chunk, page=page, threshold=threshold)
            if "entities" not in payload:
                raise PdfCloakError(
                    "ClipCloak sidecar response has no 'entities' key; "
                    "refusing to report this page clean")
            entities = payload["entities"]
            if not isinstance(entities, list):
                raise PdfCloakError("ClipCloak sidecar returned a non-list 'entities'")
            for ent in entities:
                # Fail closed: silently skipping a malformed entity could
                # be the PII that makes this page report clean.
                if not isinstance(ent, dict):
                    dropped += 1
                    continue
                label = str(ent.get("label", ""))
                if wanted and label not in wanted:
                    continue
                expected = str(ent.get("text", ""))
                try:
                    fixed = _reconcile(chunk, int(ent.get("start", 0)),
                                       int(ent.get("end", 0)), expected)
                    score = float(ent.get("score", 0.0))
                except (TypeError, ValueError):
                    dropped += 1
                    continue
                if fixed is None or not label or not expected:
                    dropped += 1
                    continue
                spans.append(Span(page, offset + fixed[0], offset + fixed[1],
                                  label, score, self.name, expected))
        if dropped:
            raise PdfCloakError(
                f"ClipCloak sidecar returned {dropped} malformed entit(ies) on "
                f"page {page + 1}; refusing to report this page clean")
        # Overlapping chunks can report the same entity twice.
        seen: set[tuple[int, int, str]] = set()
        unique: list[Span] = []
        for s in spans:
            key = (s.start, s.end, s.label)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

class OnnxDetector:
    """Generic single-graph token-classification ONNX model.

    Expects a directory with model.onnx (or *.onnx), tokenizer.json and
    config.json containing id2label with BIO/BIOES tags. GLiNER2 is *not*
    supported here — it needs 8 orchestrated graphs; use the sidecar.
    """

    name = "onnx"

    def __init__(self, model_dir: str | Path):
        try:
            import onnxruntime as ort  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            raise PdfCloakError(
                "ONNX detector needs the 'onnx' extra: uv pip install 'pdfcloak[onnx]'"
            ) from None
        d = Path(model_dir)
        models = sorted(d.glob("*.onnx"))
        if not models or not (d / "tokenizer.json").exists():
            raise PdfCloakError(f"{d} must contain an .onnx file and tokenizer.json")
        import onnxruntime as ort
        from tokenizers import Tokenizer
        try:
            self.session = ort.InferenceSession(str(models[0]), providers=["CPUExecutionProvider"])
            self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        except Exception as exc:
            raise PdfCloakError(f"cannot load ONNX model in {d}: {exc}") from None
        self.tokenizer.no_truncation()
        try:
            config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        except (OSError, ValueError) as exc:
            raise PdfCloakError(f"invalid config.json in {d}: {exc}") from None
        if not isinstance(config, dict):
            raise PdfCloakError(f"config.json in {d} must contain a JSON object")
        self.id2label: dict[int, str] = {
            int(k): v for k, v in config.get("id2label", {}).items()}
        self.input_names = [i.name for i in self.session.get_inputs()]

    MAX_TOKENS = 512
    OVERLAP_TOKENS = 64

    def _model_labels(self) -> set[str]:
        out = set()
        for tag in self.id2label.values():
            _, _, label = tag.partition("-")
            out.add((label or tag).lower())
        return out

    def detect(self, text: str, *, page: int, labels: list[str], threshold: float) -> list[Span]:
        import numpy as np
        wanted = {x.lower() for x in labels}
        if wanted and not (wanted & self._model_labels()):
            raise PdfCloakError(
                f"ONNX model labels {sorted(self._model_labels())} do not overlap "
                f"requested --labels {sorted(wanted)}; pass model label names "
                "(e.g. --labels PER,PERSON) or an empty filter via the model config")
        enc = self.tokenizer.encode(text)
        n = len(enc.ids)
        # Predict per token over overlapping windows (no silent truncation).
        pred_ids: list[int] = [-1] * n
        pred_scores: list[float] = [0.0] * n
        step = self.MAX_TOKENS - self.OVERLAP_TOKENS
        start = 0
        while True:
            end = min(start + self.MAX_TOKENS, n)
            feed: dict = {}
            arrays = {"input_ids": enc.ids[start:end],
                      "attention_mask": enc.attention_mask[start:end],
                      "token_type_ids": enc.type_ids[start:end]}
            for name in self.input_names:
                if name not in arrays:
                    raise PdfCloakError(f"unsupported ONNX input {name!r}")
                feed[name] = np.array([arrays[name]], dtype=np.int64)
            logits = self.session.run(None, feed)[0][0]
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            for j in range(end - start):
                gi = start + j
                if pred_ids[gi] == -1:
                    pred_ids[gi] = int(probs[j].argmax())
                    pred_scores[gi] = float(probs[j].max())
            if end >= n:
                break
            start += step
        spans: list[Span] = []
        cur: list | None = None  # [label, char_start, char_end, min_score]

        def flush() -> None:
            nonlocal cur
            if cur and cur[3] >= threshold:
                spans.append(Span(page, cur[1], cur[2], cur[0], round(cur[3], 4),
                                  self.name, text[cur[1]:cur[2]]))
            cur = None

        for idx in range(n):
            offset = enc.offsets[idx]
            if offset == (0, 0) or pred_ids[idx] < 0:
                continue
            tag = self.id2label.get(pred_ids[idx], "O")
            prefix, _, label = tag.partition("-")
            label = label or tag
            score = pred_scores[idx]
            if prefix in ("B", "S") or (prefix == "I" and (not cur or cur[0] != label)):
                flush()
                cur = [label, offset[0], offset[1], score]
            elif prefix in ("I", "E") and cur:
                cur[2] = offset[1]
                cur[3] = min(cur[3], score)
                if prefix == "E":
                    flush()
            else:
                flush()
        flush()
        return [s for s in spans if not wanted or s.label.lower() in wanted]
