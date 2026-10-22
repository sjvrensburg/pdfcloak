"""OCR path via ocrmypdf (external engine; tesseract under the hood).

pdfcloak itself has no OCR engine. When pages are scanned/image-only, the
supported route is: `pdfcloak ocr in.pdf -o ocr.pdf` (adds a text layer),
then scan/redact the result. ocrmypdf is looked up on PATH; a friendly,
exact error is raised when it is not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import PdfCloakError

INSTALL_HINT = (
    "ocrmypdf not found on PATH. Install it with your package manager "
    "(e.g. `brew install ocrmypdf`, `sudo apt install ocrmypdf`, or "
    "`pip install ocrmypdf` — the latter also needs tesseract), "
    "then re-run `pdfcloak ocr`.")


def ocr_available() -> str | None:
    """Path to the ocrmypdf executable, or None."""
    return shutil.which("ocrmypdf")


def run_ocr(input_path: str | Path, output_path: str | Path, *,
            lang: str = "eng", timeout: float = 3600.0,
            extra_args: list[str] | None = None) -> Path:
    """OCR `input_path` into `output_path`, adding a searchable text layer."""
    exe = ocr_available()
    if exe is None:
        raise PdfCloakError(INSTALL_HINT)
    src, dest = Path(input_path), Path(output_path)
    if not src.is_file():
        raise PdfCloakError(f"file not found: {src}")
    if dest.exists() and src.resolve() == dest.resolve():
        raise PdfCloakError("OCR output must differ from the input")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--language", lang, *(extra_args or []), str(src), str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise PdfCloakError(f"ocrmypdf timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise PdfCloakError(f"cannot run ocrmypdf: {exc}") from None
    if proc.returncode != 0 or not dest.is_file():
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise PdfCloakError(f"ocrmypdf failed (exit {proc.returncode}):\n{tail}")
    return dest
