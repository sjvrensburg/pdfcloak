"""Tiny gettext-style catalogue for CLI verdict strings (en/af/de).

Only the human-facing stderr verdicts are translated; the JSON report on
stdout keeps stable English keys/values so scripts never break. Unknown
languages and unknown keys fall back to English, never crash. Set the
language with --lang or PDFCLOAK_LANG.
"""
from __future__ import annotations

_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "found": "found: {count} match(es) ({detail})",
        "found_zero": "found: 0 matches on checked pages",
        "clean": "clean: no PII detected",
        "next": "next: {hint}",
        "unchecked": "note: {count} unchecked page(s) — result is incomplete",
    },
    "af": {
        "found": "gevind: {count} treffer(s) ({detail})",
        "found_zero": "gevind: 0 treffers op nagegane bladsye",
        "clean": "skoon: geen persoonlike inligting bespeur nie",
        "next": "volgende: {hint}",
        "unchecked": "nota: {count} onnagegane bladsy(e) — uitslag is onvolledig",
    },
    "de": {
        "found": "gefunden: {count} Treffer ({detail})",
        "found_zero": "gefunden: 0 Treffer auf geprüften Seiten",
        "clean": "sauber: keine personenbezogenen Daten gefunden",
        "next": "weiter: {hint}",
        "unchecked": "Hinweis: {count} ungeprüfte Seite(n) — Ergebnis ist unvollständig",
    },
}

SUPPORTED_LANGS = tuple(_CATALOG)


def translate(lang: str | None, key: str, **kwargs: object) -> str:
    """Translate `key` for `lang`, formatting with kwargs; en fallback.

    Region subtags are stripped (de-AT -> de, en_US -> en). Missing format
    values never leak raw {placeholders} into user-facing output.
    """
    base = (lang or "en").lower().replace("_", "-").split("-")[0]
    template = _CATALOG.get(base, _CATALOG["en"]).get(
        key, _CATALOG["en"].get(key, key))
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        import re
        return re.sub(r"\{\w+\}", "…", template)
