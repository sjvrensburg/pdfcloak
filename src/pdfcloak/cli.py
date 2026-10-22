"""pdfcloak CLI: scan | redact | verify | sidecar.

stdout stays pure JSON for scripts; all human verdicts, progress and
warnings go to stderr (suppressible with --quiet).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from . import PdfCloakError, __version__
from .detectors import DEFAULT_LABELS, ClipCloakDetector, Detector, OnnxDetector, RegexDetector
from .pages import parse_page_spec
from .i18n import SUPPORTED_LANGS, translate
from .ocr import run_ocr
from .profiles import PROFILES, get_profile
from .rules import built_in_rules, load_rules_file, parse_cli_regex
from .sidecar import (
    Sidecar, build_sidecar, bundled_sources_dir, doctor_checks,
    find_onnxruntime_lib, find_sidecar_binary, user_binary_path,
)


def _threshold(value: str) -> float:
    try:
        t = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"threshold must be a number, got {value!r}") from None
    if t != t or t in (float("inf"), float("-inf")) or not (0.0 <= t <= 1.0):
        raise argparse.ArgumentTypeError(
            f"threshold must be finite and within 0.0-1.0, got {value!r}")
    return t


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a value >= 0, got {value!r}")
    return n


def _parse_labels(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_LABELS)
    labels = [x.strip() for x in value.split(",") if x.strip()]
    if not labels:
        raise PdfCloakError(
            "--labels parsed to an empty list; omit --labels for the defaults "
            f"({','.join(DEFAULT_LABELS)})")
    return labels


def _secret(cli_value: str | None, env_arg: str | None, default_env: str,
            flag: str, *, quiet: bool = False) -> str | None:
    """Resolve a secret without putting it on the command line if avoidable."""
    if cli_value:
        if not quiet:
            print(f"warning: {flag} on the command line is visible in shell "
                  "history and process listings; prefer the env form",
                  file=sys.stderr)
        return cli_value
    env_name = env_arg or default_env
    return os.environ.get(env_name) or None


def _open_document(args):
    """Open a PDF with clean errors for corrupt/encrypted inputs."""
    import pymupdf as fitz

    path = Path(args.input)
    if not path.is_file():
        raise PdfCloakError(f"file not found: {args.input}")
    max_mb = getattr(args, "max_mb", None)
    if max_mb and path.stat().st_size > max_mb * 1024 * 1024:
        raise PdfCloakError(
            f"{args.input} is larger than --max-mb {max_mb:g} MB; "
            "raise the cap if this file is expected")
    with open(path, "rb") as fh:
        if b"%PDF-" not in fh.read(1024):
            raise PdfCloakError(
                f"{args.input} is not a PDF file (missing %PDF- header)")
    try:
        doc = fitz.open(args.input)
    except Exception as exc:
        raise PdfCloakError(f"cannot open {args.input}: {exc}") from None
    if not doc.is_pdf:
        doc.close()
        raise PdfCloakError(f"{args.input} is not a PDF file")
    max_pages = getattr(args, "max_pages", None)
    if max_pages and doc.page_count > max_pages:
        doc.close()
        raise PdfCloakError(
            f"document has {doc.page_count} pages, over --max-pages {max_pages}")
    if doc.is_encrypted or doc.needs_pass:
        password = _secret(getattr(args, "password", None),
                           getattr(args, "password_env", None),
                           "PDFCLOAK_PASSWORD", "--password",
                           quiet=getattr(args, "quiet", False))
        if not password and sys.stdin.isatty():
            password = getpass.getpass("PDF password: ")
        if not password or not doc.authenticate(password):
            doc.close()
            raise PdfCloakError(
                f"{args.input} is password-protected; pass --password, "
                "--password-env VAR, or set PDFCLOAK_PASSWORD")
    return doc


def _build_detectors(args, stack: ExitStack):
    """Return (detectors, labels, active_rule_names). Spawns the sidecar."""
    profile = get_profile(getattr(args, "profile", None))
    if getattr(args, "labels", None) is None and profile is not None:
        labels = list(profile.labels)
    elif getattr(args, "labels", None) is not None and profile is not None:
        # User labels merge with (never silently replace) profile labels.
        labels = list(dict.fromkeys(
            list(profile.labels) + _parse_labels(args.labels)))
    else:
        labels = _parse_labels(getattr(args, "labels", None))
    if getattr(args, "threshold", None) is None:
        args.threshold = profile.threshold if profile and profile.threshold else 0.55
    names = list(dict.fromkeys(getattr(args, "detector", None) or ["regex"]))
    detectors: list[Detector] = []
    custom = load_rules_file(getattr(args, "rules", None)) + parse_cli_regex(
        getattr(args, "regex", None))
    # Custom rules *add to* the built-ins; they used to silently replace
    # them, so `--regex phone:...` turned off email detection and a scan
    # could report clean while email addresses survived.
    if getattr(args, "no_builtins", False):
        rules = list(custom)
    else:
        rules = built_in_rules() + list(profile.rules if profile else []) + custom
    for name in names:
        if name == "regex":
            if not rules:
                raise PdfCloakError(
                    "no regex rules active: --no-builtins was passed but no "
                    "--regex/--rules were given")
            detectors.append(RegexDetector(rules))
        elif name == "clipcloak":
            if getattr(args, "sidecar_url", None):
                token = _secret(getattr(args, "token", None),
                                getattr(args, "token_env", None),
                                "PDFCLOAK_TOKEN", "--token",
                                quiet=getattr(args, "quiet", False))
                detectors.append(ClipCloakDetector(args.sidecar_url, token))
            else:
                sidecar = Sidecar(binary=getattr(args, "sidecar_bin", None),
                                  models_dir=getattr(args, "models_dir", None),
                                  labels=labels, threshold=args.threshold,
                                  startup_timeout=getattr(args, "sidecar_timeout", 600.0) or 600.0,
                                  log_file=getattr(args, "sidecar_log", None))
                detectors.append(stack.enter_context(sidecar))
        elif name == "onnx":
            if not getattr(args, "model_dir", None):
                raise PdfCloakError("--detector onnx requires --model-dir")
            detectors.append(OnnxDetector(args.model_dir))
    if not detectors:
        raise PdfCloakError("no detector selected")
    return detectors, labels, [r.name for r in rules]


def _report(spans, *, input_path, page_spec, scanned, show_text,
            limit: int = 0, offset: int = 0, extra: dict | None = None) -> dict:
    by_label: dict[str, int] = {}
    by_page: dict[str, int] = {}
    for s in spans:
        by_label[s.label] = by_label.get(s.label, 0) + 1
        by_page[str(s.page + 1)] = by_page.get(str(s.page + 1), 0) + 1
    page = spans[offset:offset + limit] if limit else spans[offset:]
    report = {
        "schema_version": 1,
        "input": str(input_path),
        "pages": page_spec,
        "count": len(spans),
        "returned": len(page),
        "counts_by_label": dict(sorted(by_label.items())),
        "counts_by_page": dict(sorted(by_page.items(), key=lambda kv: int(kv[0]))),
        "scanned_pages_skipped": [p + 1 for p in scanned],
        "matches": [s.report_entry(show_text=show_text) for s in page],
    }
    if extra:
        report.update(extra)
    return report


def _write_text_safely(path: str, text: str) -> None:
    """Write a report atomically with owner-only permissions (it names PII hits)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _emit_report(report: dict, path: str | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if path:
        try:
            _write_text_safely(path, text + "\n")
        except OSError as exc:
            raise PdfCloakError(f"cannot write report {path}: {exc}") from None
        print(f"report: {path} ({report['count']} match(es))", file=sys.stderr)
    else:
        print(text)


def _collapse_pages(pages: list[int]) -> str:
    """[2,6,8,9] (0-based) -> '3, 7, 9-10' for one-line warnings."""
    nums = sorted(p + 1 for p in pages)
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j > i else str(nums[i]))
        i = j + 1
    return ", ".join(parts)


def _warn_scanned(scanned: list[int], *, quiet: bool = False) -> None:
    if scanned and not quiet:
        print(f"warning: {len(scanned)} page(s) have no text layer and were "
              f"NOT checked (OCR is not yet supported): {_collapse_pages(scanned)}. "
              "To make them checkable, OCR first, e.g. `pdfcloak ocr in.pdf -o out.pdf`",
              file=sys.stderr)


def _summary(spans, scanned, *, quiet: bool, hint: str, lang: str = "en") -> None:
    if quiet:
        return
    if spans:
        by_label: dict[str, int] = {}
        for s in spans:
            by_label[s.label] = by_label.get(s.label, 0) + 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(by_label.items()))
        print(translate(lang, "found", count=len(spans), detail=detail),
              file=sys.stderr)
    else:
        print(translate(lang, "found_zero"), file=sys.stderr)
    if scanned:
        print(translate(lang, "unchecked", count=len(scanned)), file=sys.stderr)
    if hint:
        print(translate(lang, "next", hint=hint), file=sys.stderr)


def _progress(args):
    if getattr(args, "quiet", False):
        return None

    def show(_idx: int, n: int, total: int) -> None:
        if n == 1 or n == total or n % 10 == 0:
            print(f"page {n}/{total}", file=sys.stderr)

    return show


def _announce(args, detectors, rule_names) -> None:
    if getattr(args, "quiet", False):
        return
    if getattr(args, "verbose", False) or getattr(args, "regex", None) \
            or getattr(args, "rules", None):
        names = [getattr(d, "name", type(d).__name__) for d in detectors]
        print(f"detectors: {', '.join(names)}; active regex rules: "
              f"{', '.join(rule_names) or '(none)'}"
              + ("" if getattr(args, "no_builtins", False)
                 else " (built-ins included; --no-builtins to disable)"),
              file=sys.stderr)
    if getattr(args, "show_matches", False):
        print("warning: --show-matches prints matched PII text; do not share "
              "this output", file=sys.stderr)


def _fill_text(spans, page_text_list) -> None:
    page_texts = {pt.page: pt for pt in page_text_list}
    for s in spans:
        if not s.text and s.page in page_texts:
            object.__setattr__(s, "text",
                               page_texts[s.page].text[s.start:s.end])


def _pagination(args) -> tuple[int, int]:
    return getattr(args, "limit", 0) or 0, getattr(args, "offset", 0) or 0


def cmd_scan(args) -> int:
    from .redact import detect_on_pages

    with ExitStack() as stack:
        detectors, labels, rule_names = _build_detectors(args, stack)
        _announce(args, detectors, rule_names)
        with _open_document(args) as doc:
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, page_texts, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold,
                progress=_progress(args))
            _fill_text(spans, page_texts)
            _warn_scanned(scanned, quiet=args.quiet)
            limit, offset = _pagination(args)
            _emit_report(_report(spans, input_path=args.input, page_spec=args.pages,
                                 scanned=scanned, show_text=args.show_matches,
                                 limit=limit, offset=offset),
                         args.report)
            _summary(spans, scanned, quiet=args.quiet,
                     hint=f"pdfcloak redact {args.input} -o <out.pdf>", lang=getattr(args, "lang", "en") or "en")
    if getattr(args, "fail_on_findings", False) and spans:
        return 3
    if getattr(args, "fail_on_findings", False) and scanned:
        return 3
    return 0


def cmd_redact(args) -> int:
    from .redact import (
        apply_redactions, count_carriers, detect_on_pages, make_preview,
        save_safely, scrub_document, verify_output,
    )

    out_path = Path(args.output)
    if Path(args.input).resolve() == out_path.resolve():
        raise PdfCloakError("--output must differ from the input (never redact in place)")
    if out_path.is_symlink():
        raise PdfCloakError(f"output {args.output} is a symlink; refusing to follow it")
    if out_path.exists() and not args.quiet:
        print(f"note: overwriting existing output {args.output} "
              "(replaced atomically; the old file is not kept)", file=sys.stderr)
    preview_path = Path(args.preview) if getattr(args, "preview", None) else None
    with ExitStack() as stack:
        detectors, labels, rule_names = _build_detectors(args, stack)
        _announce(args, detectors, rule_names)
        with _open_document(args) as doc:
            was_encrypted = bool(doc.is_encrypted or doc.needs_pass)
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, page_text_list, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold,
                progress=_progress(args))
            page_texts = {pt.page: pt for pt in page_text_list}
            _fill_text(spans, page_text_list)
            excluded = {x.strip() for x in (args.exclude_labels or "").split(",") if x.strip()}
            retained = [s for s in spans if s.label in excluded] if excluded else []
            if excluded:
                spans = [s for s in spans if s.label not in excluded]
                if not args.quiet:
                    print(f"excluded labels: {', '.join(sorted(excluded))}",
                          file=sys.stderr)
            _warn_scanned(scanned, quiet=args.quiet)
            limit, offset = _pagination(args)
            if args.dry_run:
                extra = {"dry_run": True,
                         "scrub_preview": count_carriers(
                             doc, keep_metadata=args.keep_metadata)}
                if retained:
                    extra["retained_excluded"] = [
                        s.report_entry(show_text=args.show_matches) for s in retained]
                if preview_path is not None:
                    make_preview(doc, spans, page_texts, preview_path)
                    extra["preview"] = str(preview_path)
                report = _report(spans, input_path=args.input, page_spec=args.pages,
                                 scanned=scanned, show_text=args.show_matches,
                                 limit=limit, offset=offset, extra=extra)
                _emit_report(report, args.report)
                _summary(spans, scanned, quiet=args.quiet,
                         hint="re-run without --dry-run to write the redacted PDF",
                         lang=getattr(args, "lang", "en") or "en")
                return 0
            if getattr(args, "interactive", False) and not sys.stdin.isatty():
                raise PdfCloakError(
                    "--interactive needs a terminal to confirm on; "
                    "re-run interactively or drop --interactive")
            if getattr(args, "interactive", False) and sys.stdin.isatty():
                _summary(spans, scanned, quiet=False, hint="", lang=getattr(args, "lang", "en") or "en")
                if input(f"Redact {len(spans)} match(es)? [y/N] ").strip().lower() != "y":
                    print("aborted; nothing written", file=sys.stderr)
                    return 2
            if preview_path is not None:
                make_preview(doc, spans, page_texts, preview_path)
                if not args.quiet:
                    print(f"preview: {preview_path} (open it to inspect matches)",
                          file=sys.stderr)
            originals = [s.text for s in spans]
            apply_redactions(doc, spans, page_texts, replacement=args.replacement)
            scrub = scrub_document(doc, pages, keep_metadata=args.keep_metadata,
                                   lang=getattr(args, "lang", "en") or "en")
            save_safely(doc, args.output)
        survivors, new_hits = verify_output(
            args.output, pages, detectors, labels=labels,
            threshold=args.threshold, original_texts=originals)
        report = _report(spans, input_path=args.input, page_spec=args.pages,
                         scanned=scanned, show_text=args.show_matches,
                         limit=limit, offset=offset,
                         extra={"output": str(args.output), "scrub": scrub})
        if retained:
            report["retained_excluded"] = [
                s.report_entry(show_text=args.show_matches) for s in retained]
        report["survivors"] = [s.report_entry(show_text=args.show_matches)
                               for s in survivors]
        report["new_detections_after_redaction"] = [
            s.report_entry(show_text=args.show_matches) for s in new_hits]
        if was_encrypted:
            report["encryption_removed"] = True
            print("warning: input was encrypted; the output is NOT encrypted",
                  file=sys.stderr)
        if preview_path is not None:
            report["preview"] = str(preview_path)
        if new_hits:
            print(f"note: {len(new_hits)} fresh detector hit(s) on the redacted "
                  "text (often false positives on sentence fragments); "
                  "run `pdfcloak verify` on the output to chase them",
                  file=sys.stderr)
        if survivors:
            # Never leave a file that failed verification at the shareable
            # path where it can be mistaken for the finished deliverable.
            quarantine = Path(f"{args.output}.unsafe")
            try:
                os.replace(args.output, quarantine)
                report["quarantined_to"] = str(quarantine)
            except OSError:
                quarantine = out_path
            _emit_report(report, args.report)
            print(f"error: verification failed — {len(survivors)} match(es) "
                  f"survive; output quarantined to {quarantine}, do not share it",
                  file=sys.stderr)
            return 3
        _emit_report(report, args.report)
        if scanned and not args.allow_scanned:
            print(f"incomplete: {len(scanned)} scanned/image-only page(s) were "
                  "not inspected (no OCR); output written but NOT verified "
                  "clean — re-run with --allow-scanned to accept this",
                  file=sys.stderr)
            return 3
        if not args.quiet:
            if retained:
                print(f"redacted: {args.output} ({len(spans)} match(es) removed; "
                      f"{len(retained)} retained by --exclude-labels — "
                      "output intentionally still contains those, not clean)",
                      file=sys.stderr)
            else:
                print(f"redacted: {args.output} ({len(spans)} match(es), verified clean)",
                      file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    from .redact import detect_on_pages

    with ExitStack() as stack:
        detectors, labels, rule_names = _build_detectors(args, stack)
        _announce(args, detectors, rule_names)
        with _open_document(args) as doc:
            pages = parse_page_spec(args.pages, doc.page_count)
            spans, page_texts, scanned = detect_on_pages(
                doc, pages, detectors, labels=labels, threshold=args.threshold,
                progress=_progress(args))
            _fill_text(spans, page_texts)
            _warn_scanned(scanned, quiet=args.quiet)
            limit, offset = _pagination(args)
            _emit_report(_report(spans, input_path=args.input, page_spec=args.pages,
                                 scanned=scanned, show_text=args.show_matches,
                                 limit=limit, offset=offset),
                         args.report)
    if spans:
        print(f"{len(spans)} PII match(es) still present", file=sys.stderr)
        return 3
    if scanned:
        print(f"incomplete: {len(scanned)} scanned/image-only page(s) could "
              "not be checked (no OCR); not verified clean", file=sys.stderr)
        return 3
    print(translate(getattr(args, "lang", "en") or "en", "clean"), file=sys.stderr)
    return 0



def cmd_ocr(args) -> int:
    out = run_ocr(args.input, args.output, lang=getattr(args, "ocr_lang", "eng") or "eng")
    if not args.quiet:
        print(f"ocr: {out} (text layer added; now run scan/redact on it)",
              file=sys.stderr)
    return 0


def _batch_files(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(f for f in directory.glob(pattern) if f.is_file())


def cmd_batch(args) -> int:
    """Run scan/redact over every PDF in a directory; one JSON summary."""
    directory = Path(args.input)
    if not directory.is_dir():
        raise PdfCloakError(f"batch input is not a directory: {args.input}")
    files = _batch_files(directory, getattr(args, "recursive", False))
    out_dir: Path | None = None
    if args.action == "redact":
        if not getattr(args, "output_dir", None):
            raise PdfCloakError("batch redact requires -o/--output-dir")
        out_dir = Path(args.output_dir)
        resolved_out, resolved_in = out_dir.resolve(), directory.resolve()
        if resolved_out == resolved_in or resolved_in in resolved_out.parents:
            raise PdfCloakError(
                "batch output directory must not be the input directory or "
                "inside it (outputs would overwrite/scatter into the inputs)")
    results: list[dict] = []
    worst = 0
    with ExitStack() as stack:
        detectors, labels, rule_names = _build_detectors(args, stack)
        _announce(args, detectors, rule_names)
        for pdf in files:
            entry: dict = {"input": str(pdf)}
            try:
                file_args = argparse.Namespace(**{**vars(args), "input": str(pdf)})
                with _open_document(file_args) as doc:
                    from .redact import detect_on_pages
                    pages = parse_page_spec(args.pages, doc.page_count)
                    spans, page_texts, scanned = detect_on_pages(
                        doc, pages, detectors, labels=labels,
                        threshold=args.threshold)
                    _fill_text(spans, page_texts)
                    entry.update({"count": len(spans),
                                  "counts_by_label": _report(
                                      spans, input_path=pdf, page_spec=args.pages,
                                      scanned=scanned, show_text=False)["counts_by_label"],
                                  "unchecked_pages": len(scanned)})
                    if args.action == "redact":
                        from .redact import (apply_redactions, save_safely,
                                             scrub_document, verify_output)
                        assert out_dir is not None
                        # Mirror relative paths: a/x.pdf and b/x.pdf must
                        # not flatten onto one output and overwrite it.
                        out_path = out_dir / pdf.relative_to(directory)
                        if out_path.resolve() == pdf.resolve():
                            raise PdfCloakError(
                                f"batch output would overwrite input {pdf}")
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        by_page = {pt.page: pt for pt in page_texts}
                        originals = [s.text for s in spans]
                        apply_redactions(doc, spans, by_page,
                                         replacement=args.replacement)
                        scrub_document(doc, pages,
                                       keep_metadata=args.keep_metadata,
                                       lang=getattr(args, "lang", None) or "en")
                        save_safely(doc, out_path)
                        entry["output"] = str(out_path)
                        survivors, _new = verify_output(
                            out_path, pages, detectors, labels=labels,
                            threshold=args.threshold, original_texts=originals)
                        if survivors:
                            quarantine = Path(f"{out_path}.unsafe")
                            os.replace(out_path, quarantine)
                            entry["quarantined_to"] = str(quarantine)
                            entry["rc"] = 3
                        else:
                            entry["rc"] = 3 if (scanned and not args.allow_scanned) else 0
                    else:
                        entry["rc"] = 3 if spans or scanned else 0
            except Exception as exc:  # per-file: record and keep going
                entry.update({"rc": 2,
                              "error": str(exc) if isinstance(exc, PdfCloakError)
                              else f"{exc.__class__.__name__}: {exc}"})
            results.append(entry)
            worst = max(worst, int(entry.get("rc", 0)))
    summary = {"action": args.action, "directory": str(directory),
               "files": results,
               "total_matches": sum(r.get("count", 0) for r in results)}
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if getattr(args, "report", None):
        _write_text_safely(args.report, text + "\n")
        print(f"report: {args.report}", file=sys.stderr)
    else:
        print(text)
    return worst


def cmd_serve(args) -> int:
    from .web import serve
    server = serve(args.host, args.port)
    host, port = server.server_address[:2]
    host = host.decode() if isinstance(host, bytes) else str(host)
    print(f"pdfcloak UI: http://{host}:{port}/ (Ctrl+C to stop; "
          "loopback only, no authentication)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
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
              "Hugging Face (cached afterwards). Tip: run "
              "`pdfcloak sidecar doctor` to preflight everything.",
              file=sys.stderr)
        return 0
    if args.action == "doctor":
        rows = doctor_checks()
        binary_ok = rows[0][1]
        failed = False
        for name, ok, fix in rows:
            if name.startswith("cargo") and binary_ok:
                continue  # cargo only matters when there is no binary to run
            print(f"{'ok  ' if ok else 'MISS'} {name}" + ("" if ok else f" — fix: {fix}"))
            if not ok and name in ("sidecar binary", "onnxruntime library"):
                failed = True
        if failed:
            print("doctor: not ready for AI detection (regex detection still works)",
                  file=sys.stderr)
            return 3
        print("doctor: ready", file=sys.stderr)
        return 0
    # status (informational; always exit 0 — use `doctor` for a gate)
    binary = find_sidecar_binary()
    print(f"binary:   {binary or 'not found (run: pdfcloak sidecar build)'}")
    print(f"expected: {user_binary_path()}")
    print(f"sources:  {bundled_sources_dir() or 'not bundled'}")
    print(f"onnxruntime lib: {find_onnxruntime_lib() or 'not found'}")
    return 0


_SCAN_EXAMPLES = """examples:
  pdfcloak scan in.pdf
  pdfcloak scan in.pdf --pages 1,3-5 --rules rules.toml --fail-on-findings
  pdfcloak scan in.pdf --regex phone:'\\+?[0-9][0-9 ]{7,}' --no-builtins
"""
_REDACT_EXAMPLES = """examples:
  pdfcloak redact in.pdf -o out.pdf
  pdfcloak redact in.pdf -o out.pdf --dry-run --preview preview.pdf
  pdfcloak redact in.pdf -o out.pdf --replacement label --exclude-labels url
"""
_VERIFY_EXAMPLES = """examples:
  pdfcloak verify out.pdf
  pdfcloak verify out.pdf --detector clipcloak --labels name,email
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdfcloak",
                                description="Redact PII from PDF documents.")
    p.add_argument("--version", action="version", version=f"pdfcloak {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("input", help="input PDF")
        det = sp.add_argument_group("detection")
        det.add_argument("--pages", default="all",
                         help="1-based spec: all | 3 | 3-7 | 3- | -7 | 1,4,last")
        det.add_argument("--rules", help="TOML file with [[rule]] name/pattern entries")
        det.add_argument("--regex", action="append", metavar="NAME:PATTERN",
                         help="inline regex rule (repeatable; adds to built-ins)")
        det.add_argument("--no-builtins", action="store_true",
                         help="disable built-in email/url/ipv4 rules")
        det.add_argument("--detector", action="append",
                         choices=["regex", "clipcloak", "onnx"],
                         help="detector to enable (repeatable; default: regex)")
        det.add_argument("--labels", help="comma-separated model labels")
        det.add_argument("--profile", choices=sorted(PROFILES),
                         help="policy profile: label/rule bundle (gdpr, hipaa, legal, sa, minimal)")
        det.add_argument("--lang", default=None, choices=list(SUPPORTED_LANGS),
                         help="language for messages (and output /Lang); default: en or $PDFCLOAK_LANG")
        det.add_argument("--threshold", type=_threshold, default=None,
                         help="model score threshold 0.0-1.0 (regex ignores it; "
                              "default 0.55 or the profile's)")
        model = sp.add_argument_group("model & sidecar (advanced)")
        model.add_argument("--password", help="password for encrypted PDFs "
                           "(prefer --password-env / PDFCLOAK_PASSWORD)")
        model.add_argument("--password-env", metavar="VAR",
                           help="read the PDF password from env var VAR")
        model.add_argument("--sidecar-url", help="URL of a running ClipCloak daemon "
                           "(loopback, or https with PDFCLOAK_ALLOW_REMOTE_SIDECAR=1)")
        model.add_argument("--sidecar-bin", help="path to clipcloak-server binary")
        model.add_argument("--sidecar-log", help="write sidecar stdout/stderr to this file")
        model.add_argument("--sidecar-timeout", type=float, default=600.0,
                           help="seconds to wait for the sidecar to become healthy")
        model.add_argument("--token", help="Bearer [REDACTED] for --sidecar-url "
                           "(prefer --token-env / PDFCLOAK_TOKEN)")
        model.add_argument("--token-env", metavar="VAR",
                           help="read the sidecar token from env var VAR")
        model.add_argument("--models-dir", help="local GLiNER2 fragments (sidecar)")
        model.add_argument("--model-dir", help="local single-graph ONNX model dir")
        rep = sp.add_argument_group("report & output")
        rep.add_argument("--report", help="write JSON report to this file")
        rep.add_argument("--show-matches", action="store_true",
                         help="include matched PII text in the report (sensitive!)")
        rep.add_argument("--limit", type=_positive_int, default=0,
                         help="max matches to include in the report (0 = all)")
        rep.add_argument("--offset", type=_positive_int, default=0,
                         help="skip this many matches in the report")
        rep.add_argument("--quiet", action="store_true",
                         help="suppress progress, summaries and warnings on stderr")
        rep.add_argument("--verbose", action="store_true",
                         help="print detectors and active rules on stderr")
        lim = sp.add_argument_group("limits")
        lim.add_argument("--max-pages", type=_positive_int, default=0,
                         help="refuse documents with more pages (0 = no limit)")
        lim.add_argument("--max-mb", type=float, default=0,
                         help="refuse input files larger than this many MB (0 = no limit)")

    sp = sub.add_parser("scan", help="detect only; never modifies the PDF",
                        epilog=_SCAN_EXAMPLES,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(sp)
    sp.add_argument("--fail-on-findings", action="store_true",
                    help="exit 3 when matches (or unchecked pages) are found")
    sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("redact", help="detect, redact, scrub, save, verify",
                        epilog=_REDACT_EXAMPLES,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(sp)
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--force", action="store_true",
                    help="accepted for compatibility; existing outputs are replaced atomically with a note")
    sp.add_argument("--replacement", choices=["black", "label"], default="black")
    sp.add_argument("--allow-scanned", action="store_true",
                    help="accept exit 0 although scanned pages were skipped")
    sp.add_argument("--keep-metadata", action="store_true")
    sp.add_argument("--exclude-labels",
                    help="comma-separated match labels to leave in place")
    sp.add_argument("--preview",
                    help="also write an annotated preview PDF of the matches")
    sp.add_argument("--interactive", action="store_true",
                    help="ask for confirmation on a terminal before redacting")
    sp.add_argument("--dry-run", action="store_true",
                    help="report matches without writing a redacted PDF")
    sp.set_defaults(func=cmd_redact)
    sp = sub.add_parser("verify", help="re-check a redacted PDF (exit 3 if PII remains)",
                        epilog=_VERIFY_EXAMPLES,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(sp)
    sp.set_defaults(func=cmd_verify)
    sp = sub.add_parser("ocr", help="add a text layer to a scanned PDF (needs ocrmypdf)")
    sp.add_argument("input", help="input (scanned) PDF")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--ocr-lang", default="eng",
                    help="tesseract language code(s), e.g. eng, afr, deu (default: eng)")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_ocr)
    sp = sub.add_parser("batch", help="scan/redact every PDF in a directory")
    sp.add_argument("action", choices=["scan", "redact"])
    sp.add_argument("input", help="directory containing PDFs")
    sp.add_argument("-o", "--output-dir", help="output directory (batch redact)")
    sp.add_argument("--recursive", action="store_true")
    sp.add_argument("--pages", default="all")
    sp.add_argument("--rules", help="TOML file with [[rule]] name/pattern entries")
    sp.add_argument("--regex", action="append", metavar="NAME:PATTERN")
    sp.add_argument("--no-builtins", action="store_true")
    sp.add_argument("--profile", choices=sorted(PROFILES))
    sp.add_argument("--lang", default=None, choices=list(SUPPORTED_LANGS))
    sp.add_argument("--detector", action="append", choices=["regex", "clipcloak", "onnx"])
    sp.add_argument("--labels")
    sp.add_argument("--threshold", type=_threshold, default=None)
    sp.add_argument("--password")
    sp.add_argument("--password-env", metavar="VAR")
    sp.add_argument("--sidecar-url")
    sp.add_argument("--sidecar-bin")
    sp.add_argument("--sidecar-log")
    sp.add_argument("--token")
    sp.add_argument("--token-env", metavar="VAR")
    sp.add_argument("--models-dir")
    sp.add_argument("--model-dir")
    sp.add_argument("--sidecar-timeout", type=float, default=600.0)
    sp.add_argument("--replacement", choices=["black", "label"], default="black")
    sp.add_argument("--allow-scanned", action="store_true")
    sp.add_argument("--keep-metadata", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--report", help="write the batch JSON summary to this file")
    sp.add_argument("--max-pages", type=_positive_int, default=0)
    sp.add_argument("--max-mb", type=float, default=0)
    sp.set_defaults(func=cmd_batch)
    sp = sub.add_parser("serve", help="local web UI on loopback (no authentication)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("sidecar", help="manage the ClipCloak GLiNER2 sidecar binary")
    sp.add_argument("action", choices=["build", "status", "doctor"],
                    help="doctor = preflight checks with a nonzero exit when not ready")
    sp.add_argument("--force", action="store_true", help="rebuild even if present")
    sp.set_defaults(func=cmd_sidecar)
    return p


def _resolve_lang(args) -> None:
    """args.lang (--lang) > PDFCLOAK_LANG > 'en'; region subtags stripped."""
    lang = getattr(args, "lang", None) or os.environ.get("PDFCLOAK_LANG") or "en"
    base = str(lang).lower().replace("_", "-").split("-")[0]
    args.lang = base if base in SUPPORTED_LANGS else "en"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_lang(args)
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
