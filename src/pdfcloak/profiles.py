"""Policy profiles: named bundles of labels, extra rules, and thresholds.

A profile is a starting point, not magic compliance: it tunes what the
detectors look for for a regulatory context. The user's own --labels /
--regex / --rules and the built-in rules are always merged on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rules import Rule


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    threshold: float | None = None


_BASE_LABELS = ["name", "street address", "email", "phone_num", "id_num",
                "url", "username"]

PROFILES: dict[str, Profile] = {
    "gdpr": Profile(
        "gdpr",
        "EU/UK GDPR personal data: identifiers, contact details, IBANs.",
        labels=_BASE_LABELS + ["date of birth"],
        rules=[Rule("iban", r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,6}\b"),
               Rule("eu_vat", r"\b[A-Z]{2}\d{8,12}\b")],
    ),
    "hipaa": Profile(
        "hipaa",
        "US HIPAA PHI: SSNs, medical record numbers, dates, contact details.",
        labels=_BASE_LABELS + ["date of birth", "medical record number"],
        rules=[Rule("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
               Rule("mrn", r"\bMRN[:# ]?\d{5,10}\b")],
        threshold=0.5,
    ),
    "legal": Profile(
        "legal",
        "Legal documents: parties, case numbers, ID numbers, contact details.",
        labels=["name", "street address", "email", "phone_num", "id_num"],
        rules=[Rule("case_number", r"\b(?:Case|Matter|File)[:# ]?[A-Z0-9-]{4,}\b")],
    ),
    "sa": Profile(
        "sa",
        "South African documents: SA ID numbers, local phone formats.",
        labels=list(_BASE_LABELS),
        rules=[Rule("sa_id", r"\b\d{13}\b"),
               Rule("sa_phone", r"\b(?:\+27|0)\d{2}[ ]?\d{3}[ ]?\d{4}\b")],
    ),
    "minimal": Profile(
        "minimal",
        "Only structured identifiers (email/url/ipv4 built-ins); no model labels.",
        labels=[],
        rules=[],
    ),
}


def get_profile(name: str | None) -> Profile | None:
    if name is None:
        return None
    profile = PROFILES.get(name.lower())
    if profile is None:
        from . import PdfCloakError
        raise PdfCloakError(
            f"unknown --profile {name!r}; choose from {', '.join(sorted(PROFILES))}")
    return profile
