"""Named regex rules (TOML file and/or --regex name:pattern).

Regex matches get score 2.0 so they always outrank model spans.
Uses the third-party `regex` module, which supports per-call timeouts
(stdlib `re` can catastrophically backtrack).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import regex as regex_module

from . import PdfCloakError

RULE_SCORE = 2.0
MAX_MATCHES_PER_RULE = 512
MATCH_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str

    def compile(self) -> "regex_module.Pattern[str]":
        try:
            return regex_module.compile(self.pattern)
        except regex_module.error as exc:
            raise PdfCloakError(f"rule {self.name!r}: invalid regex: {exc}") from None


def parse_cli_regex(items: list[str] | None) -> list[Rule]:
    rules: list[Rule] = []
    for item in items or []:
        name, sep, pattern = item.partition(":")
        if not sep or not name.strip() or not pattern:
            raise PdfCloakError(f"--regex expects name:pattern, got {item!r}")
        rules.append(Rule(name.strip(), pattern))
    return rules


def load_rules_file(path: str | Path | None) -> list[Rule]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise PdfCloakError(f"rules file not found: {p}")
    try:
        raw_text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PdfCloakError(f"cannot read rules file {p}: {exc}") from None
    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise PdfCloakError(f"invalid TOML in {p}: {exc}") from None
    entries = data.get("rule", [])
    if not isinstance(entries, list):
        raise PdfCloakError(f"{p}: 'rule' must be an array of [[rule]] tables")
    rules: list[Rule] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PdfCloakError(f"{p}: rule #{i} must be a [[rule]] table")
        if entry.get("enabled", True) is False:
            continue
        name, pattern = entry.get("name"), entry.get("pattern")
        if not isinstance(name, str) or not isinstance(pattern, str) \
                or not name.strip() or not pattern:
            raise PdfCloakError(
                f"{p}: rule #{i} needs string name and pattern fields")
        rules.append(Rule(name.strip(), pattern))
    return rules


def built_in_rules() -> list[Rule]:
    """Sensible defaults. Always active unless --no-builtins is passed.

    The email pattern is Unicode-aware (the `regex` module's ``\\w`` covers
    accented/Cyrillic/CJK local parts and IDN domains), so non-English
    addresses are not silently missed by an ASCII-only character class.
    """
    return [
        Rule("email", r"[\w.+-]+@[^\W_]+(?:-[^\W_]+)*(?:\.[^\W_]+(?:-[^\W_]+)*)*\.[A-Za-z]{2,}"),
        Rule("url", r"https?://[^\s<>()\[\]]+"),
        Rule("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ]


def compile_rules(rules: list[Rule]) -> list[tuple[Rule, "regex_module.Pattern[str]"]]:
    """Compile eagerly so a bad pattern fails at startup, never silently."""
    return [(rule, rule.compile()) for rule in rules]
