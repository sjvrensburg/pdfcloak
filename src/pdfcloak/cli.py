"""pdfcloak CLI: scan | redact | verify."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import pymupdf as fitz

from . import PdfCloakError, __version__
from .detectors import DEFAULT_LABELS, ClipCloakDetector, OnnxDetector, RegexDetector
from .pages import parse_page_spec
from .redact import apply_redactions, detect_on_pages, save_safely, scrub_document, verify_output
from .rules import built_in_rules, load_rules_file, parse_cli_regex
from .sidecar import Sidecar, build_sidecar, bundled_sources_dir, find_onnxruntime_lib, find_sidecar_binary, user_binary_path


def _threshold(value: str) -> float:
    try:
        t = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"threshold must be a number, got {value!r}") from None
    if t != t or t in (float("inf"), float("-inf")) or not (0.0 <= t <= 1.0):
        raise argparse.ArgumentTypeError(
            f"threshold must be finite and within 0.0-1.0, got {value!r}")
    return t


def _parse_labels(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_LABELS)
    labels = [x.strip() for x in value.split(",") if x.strip()]
    if not labels:
        raise PdfCloakError(
            "--labels parsed to an empty list; omit --labels for the defaults "
            f"({','.join(DEFAULT_LABELS)})")
    return labels


def _open_document(args) -> fitz.Document:
    """Open a PDF with clean errors for corrupt/encrypted inputs."""
    try:
        doc = fitz.open(args.input)
    except FileNotFoundError:
        raise PdfCloakError(f"file not found: {args.input}") from None
    except Exception as exc:
        raise PdfCloakError(f"cannot open {args.input}: {exc}") from None
    if doc.is_encrypted or doc.needs_pass:
        password = getattr(args, "password", None)
        if not password or not doc.authenticate(password):
            doc.close()
            raise PdfCloakError(
                f"{args.input} is password-protected; pass --password to open it")
    return doc


def _build_detectors(args, stack: ExitStack):
    """Return (detectors, labels). Spawns the sidecar when requested."""
    labels = _parse_labels(getattr(args, "labels", None))
    names = list(dict.fromkeys(getattr(args, "detector", None) or ["regex"]))
    detectors = []
    rules = load_rules_file(getattr(args, "rules", None)) + parse_cli_regex(
        getattr(args, "regex", None))
    for name in names:
        if name == "regex":
            detectors.append(RegexDetector(rules or built_in_rules()))
        elif name == "clipcloak":
            if getattr(args, "sidecar_url", None):
                detectors.append(ClipCloakDetector(args.sidecar_url, args.token))
            else:
                sidecar = Sidecar(binary=getattr(args, "sidecar_bin", None),
                                  models_dir=getattr(args, "models_dir", None),
                                  labels=labels, threshold=args.threshold,
                                  log_file=getattr(args, "sidecar_log", None))
                detectors.append(stack.enter_context(sidecar))
        elif name == "onnx":
            if not getattr(args, "model_dir", None):
                raise PdfCloakError("--detector onnx requires --model-dir")
            detectors.append(OnnxDetector(args.model_dir))
    if not detectors:
        raise PdfCloakError("no detector selected")
    return detectors, labels


def _report(spans, *, input_path, page_spec, scanned, show_text) -> dict:
    return {
        "input": str(input_path),
        "pages": page_spec,
        "count": len(spans),
        "scanned_pages_skipped": [p + 1 for p in scanned],
        "matches": [s.report_entry(show_text=show_text) for s in spans],
    }


def _emit_report(report: dict, path: str | None, show_text: bool) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if path:
        try:
            Path(path).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise PdfCloakError(f"cannot write report {path}: {exc}") from None
        print(f"report: {path} ({report['count']} match(es))", file=sys.stderr)
    else:
        print(text)


def _warn_scanned(scanned: list[int]) -> None:
    for p in scanned:
        print(f"warning: page {p + 1} has no text layer (scanned/image-only); "
              "skipped — OCR is not yet supported", file=sys.stderr)


def cmd_scan(args) -> int:
    with ExitStack() as stack:
        detectors, labels = _build_detectors(args, stack)
        with _open_document(args) as doc:
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, page_texts, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold)
            for pt in page_texts:
                for s in spans:
                    if s.page == pt.page and not s.text:
                        object.__setattr__(s, "text", pt.text[s.start:s.end])
            _warn_scanned(scanned)
            _emit_report(_report(spans, input_path=args.input, page_spec=args.pages,
                                 scanned=scanned, show_text=args.show_matches),
                         args.report, args.show_matches)
    return 0


def cmd_redact(args) -> int:
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise PdfCloakError("--output must differ from the input (never redact in place)")
    with ExitStack() as stack:
        detectors, labels = _build_detectors(args, stack)
        with _open_document(args) as doc:
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, page_text_list, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold)
            page_texts = {pt.page: pt for pt in page_text_list}
            for s in spans:
                if not s.text:
                    object.__setattr__(s, "text",
                                       page_texts[s.page].text[s.start:s.end])
            _warn_scanned(scanned)
            report = _report(spans, input_path=args.input, page_spec=args.pages,
                             scanned=scanned, show_text=args.show_matches)
            if args.dry_run:
                report["dry_run"] = True
                _emit_report(report, args.report, args.show_matches)
                return 0
            originals = [s.text for s in spans]
            apply_redactions(doc, spans, page_texts, replacement=args.replacement)
            scrub_document(doc, pages, keep_metadata=args.keep_metadata)
            save_safely(doc, args.output)
        survivors, new_hits = verify_output(
            args.output, pages, detectors, labels=labels,
            threshold=args.threshold, original_texts=originals)
        report["output"] = str(args.output)
        report["survivors"] = [s.report_entry(show_text=args.show_matches)
                                for s in survivors]
        report["new_detections_after_redaction"] = [
            s.report_entry(show_text=args.show_matches) for s in new_hits]
        if new_hits:
            print(f"note: {len(new_hits)} fresh detector hit(s) on the redacted "
                  "text (often false positives on sentence fragments); "
                  "run `pdfcloak verify` on the output to chase them",
                  file=sys.stderr)
        _emit_report(report, args.report, args.show_matches)
        if survivors:
            print(f"error: verification failed — {len(survivors)} match(es) "
                  f"survive in {args.output}; do not share this file", file=sys.stderr)
            return 3
        if scanned and not args.allow_scanned:
            print(f"incomplete: {len(scanned)} scanned/image-only page(s) were "
                  "not inspected (no OCR); output written but NOT verified "
                  "clean — re-run with --allow-scanned to accept this",
                  file=sys.stderr)
            return 3
        print(f"redacted: {args.output} ({len(spans)} match(es), verified clean)",
              file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    with ExitStack() as stack:
        detectors, labels = _build_detectors(args, stack)
        with _open_document(args) as doc:
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, _, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold)
            _warn_scanned(scanned)
            _emit_report(_report(spans, input_path=args.input, page_spec=args.pages,
                                 scanned=scanned, show_text=args.show_matches),
                         args.report, args.show_matches)
    if spans:
        print(f"{len(spans)} PII match(es) still present", file=sys.stderr)
        return 3
    if scanned:
        print(f"incomplete: {len(scanned)} scanned/image-only page(s) could "
              "not be checked (no OCR); not verified clean", file=sys.stderr)
        return 3
    print("clean: no PII detected", file=sys.stderr)
    return 0


def cmd_sidecar(args) -> int:
    if args.action == "build":
        path = build_sidecar(force=args.force)
        print(f"sidecar binary: {path}")
        if not find_onnxruntime_lib():
            print("note: no libonnxruntime.so found; install the onnx extra "
                  "into the tool env, e.g. `uv tool install --with onnxruntime "
                  "pdfcloak` / `pipx inject pdfcloak onnxruntime`, or set "
                  "ORT_DYLIB_PATH", file=sys.stderr)
        print("first AI run downloads ~412 MB of GLiNER2 weights from "
              "Hugging Face (cached afterwards)", file=sys.stderr)
        return 0
    # status
    binary = find_sidecar_binary()
    print(f"binary:   {binary or 'not found (run: pdfcloak sidecar build)'}")
    print(f"expected: {user_binary_path()}")
    print(f"sources:  {bundled_sources_dir() or 'not bundled'}")
    print(f"onnxruntime lib: {find_onnxruntime_lib() or 'not found'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdfcloak",
                                description="Redact PII from PDF documents.")
    p.add_argument("--version", action="version", version=f"pdfcloak {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("input", help="input PDF")
        sp.add_argument("--pages", default="all",
                        help="1-based spec: all | 3 | 3-7 | 3- | -7 | 1,4,last")
        sp.add_argument("--rules", help="TOML file with [[rule]] name/pattern entries")
        sp.add_argument("--regex", action="append", metavar="NAME:PATTERN",
                        help="inline regex rule (repeatable)")
        sp.add_argument("--detector", action="append",
                        choices=["regex", "clipcloak", "onnx"],
                        help="detector to enable (repeatable; default: regex)")
        sp.add_argument("--labels", help="comma-separated model labels")
        sp.add_argument("--threshold", type=_threshold, default=0.55,
                        help="model score threshold 0.0-1.0 (regex ignores it)")
        sp.add_argument("--password", help="password for encrypted PDFs")
        sp.add_argument("--sidecar-url", help="URL of a running ClipCloak daemon")
        sp.add_argument("--sidecar-bin", help="path to clipcloak-server binary")
        sp.add_argument("--sidecar-log", help="write sidecar stdout/stderr to this file")
        sp.add_argument("--token", help="bearer token for --sidecar-url")
        sp.add_argument("--models-dir", help="local GLiNER2 fragments (sidecar) ")
        sp.add_argument("--model-dir", help="local single-graph ONNX model dir")
        sp.add_argument("--report", help="write JSON report to this file")
        sp.add_argument("--show-matches", action="store_true",
                        help="include matched PII text in the report (sensitive!)")

    sp = sub.add_parser("scan", help="detect only; never modifies the PDF")
    common(sp); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("redact", help="detect, redact, scrub, save, verify")
    common(sp)
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--replacement", choices=["black", "label"], default="black")
    sp.add_argument("--allow-scanned", action="store_true",
                    help="accept exit 0 although scanned pages were skipped")
    sp.add_argument("--keep-metadata", action="store_true")
    sp.add_argument("--dry-run", action="store_true",
                    help="report matches without writing a redacted PDF")
    sp.set_defaults(func=cmd_redact)
    sp = sub.add_parser("verify", help="re-check a redacted PDF (exit 3 if PII remains)")
    common(sp); sp.set_defaults(func=cmd_verify)
    sp = sub.add_parser("sidecar", help="manage the ClipCloak GLiNER2 sidecar binary")
    sp.add_argument("action", choices=["build", "status"])
    sp.add_argument("--force", action="store_true", help="rebuild even if present")
    sp.set_defaults(func=cmd_sidecar)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PdfCloakError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # corrupt PDFs, unwritable outputs, bad sidecar data
        print(f"error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
