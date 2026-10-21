"""pdfcloak — redact PII from PDFs. AGPL-3.0-or-later."""

__version__ = "0.1.0"


class PdfCloakError(Exception):
    """User-facing error (bad input, detector failure, verification failure)."""
