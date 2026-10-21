# pdfcloak

CLI tool that redacts PII from PDF documents. Regex rules plus an on-device
GLiNER2 AI detector, physically removing text with PyMuPDF — never drawing
black boxes over it. AGPL-3.0-or-later.

See [pdf-editor-redaction-report.md](pdf-editor-redaction-report.md) for the
research behind these choices.

## Install

As a tool (recommended). From a checkout of this repository:

```bash
uv tool install .          # or: pipx install .
pdfcloak scan in.pdf
```

One-off, without installing:

```bash
uvx --from . pdfcloak scan in.pdf
pipx run --spec . pdfcloak scan in.pdf
```

Once published to PyPI the same commands work with just `pdfcloak` in place
of `.` (`uv tool install pdfcloak`, `uvx pdfcloak …`).

For development: `uv sync --group dev && uv run pdfcloak …`.

## Quick start

```bash
pdfcloak scan in.pdf
pdfcloak redact in.pdf -o out.pdf --regex email:'[A-Za-z0-9._%+-]+@[^ ]+'
pdfcloak verify out.pdf
```

## AI detector (sidecar, Option 2 distribution)

The GLiNER2 detector ships as a vendored Rust sidecar
(`vendor/clipcloak-server/`, originally ClipCloak). pdfcloak spawns it,
talks to it over localhost HTTP, and kills it on exit:

The wheel bundles the sidecar's Rust sources, so an installed tool can build
its own binary. This is the only step that needs a Rust toolchain
(https://rustup.rs); regex-only use needs no sidecar at all.

```bash
pdfcloak sidecar build     # one-time cargo build -> ~/.local/share/pdfcloak/bin/
pdfcloak sidecar status    # check binary / sources / libonnxruntime locations
pdfcloak redact in.pdf -o out.pdf --detector clipcloak
```

`libonnxruntime.so` must be visible to the sidecar: it is found automatically
when `onnxruntime` is in the tool environment (add it with
`uv tool install --with onnxruntime .` or `pipx inject pdfcloak onnxruntime`
if missing), or set `ORT_DYLIB_PATH`. In a repo checkout,
`scripts/build-sidecar.sh` still builds into `vendor/…/target/release/`.

First sidecar run downloads ~412 MB of int8 model weights from Hugging Face
(`stefanj0/gliner2-sa-names-lora`, `onnx_int8`), unless `--models-dir` points
at local ONNX fragments. To use an already-running daemon instead:
`--sidecar-url http://127.0.0.1:8731 --token <PII_TOKEN>`.

The vendored copy carries two pdfcloak fixes (see
`vendor/clipcloak-server/UPSTREAM.md`): upstream routed the `onnx_int8`
subfolder to the wrong engine and picked `_fp32.onnx` filenames that do not
exist, so a fresh sidecar died with a 404. The local-dir path was unaffected.
Noisy `libonnxruntime_providers_openvino.so` errors in the sidecar log are
harmless — that execution provider is absent and inference falls back to CPU.

`redact` verifies its own output: it fails (exit 3) only if an originally
matched string survives. Fresh detector hits on the mangled text are reported
as `new_detections_after_redaction` without failing — the model sometimes
flags sentence fragments (e.g. "at") after neighbours are removed. Run
`pdfcloak verify <output>` if you want to chase those too.

## Commands

- `scan` — detect only, JSON report, never modifies the PDF.
- `redact` — detect, redact, scrub metadata/attachments, save, verify.
- `verify` — re-check a redacted PDF; exit 3 if PII is still detected.

Pages are 1-based: `--pages 1,3-5,last`, `3-`, `-7`, `all`.
Rules come from `--regex name:pattern` (repeatable) or a TOML `--rules` file
with `[[rule]]` entries. Regex matches outrank model matches.
Matched PII is hashed (SHA-256 prefix) in reports, never printed, unless
`--show-matches` is given.

Exit codes: 0 = clean/done, 2 = usage/input error (corrupt, encrypted
without `--password`, bad rules, missing file — always a clean one-line
error, never a traceback), 3 = PII found / survivors / incomplete.

Scrubbing removes annotations, form widgets, link targets and bookmarks on
**all** pages, plus document metadata/XMP and attachments: those are
invisible PII carriers. Visible text on pages outside `--pages` is left
alone, as page scoping intends. Encrypted PDFs open with `--password`.

Reading order ports the heuristics from
[RailReaderCore](https://github.com/sjvrensburg/RailReaderCore) (see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)): lines are first grouped
into prose blocks with Docstrum-style thresholds taken from the page's own
line pitch, then blocks are ordered with XY-Cut++ — validated column
gutters (coverage/width floors, 7pt minimum), a projection-profile gutter
fallback that tolerates a block crossing the gutter, full-width divider
handling, and margin-note masking with nearest-neighbour reinsertion.
Two-column layouts therefore extract column-by-column instead of across
the gutter. Sideways/rotated text and complex table layouts remain
best-effort. The PP-DocLayout ONNX layout models RailReader also uses were
deliberately not ported: GLiNER2 already supplies this project's model,
and block *classes* (figure/table/heading) do not change redaction order.

Known limits: no OCR yet — scanned/image-only pages are warned about,
listed in the report, and make `verify` and `redact` exit 3 (override with
`--allow-scanned`). Detection also runs on a line-joined view so tokens
wrapped across lines match, but matches cannot cross page boundaries.
Model weights are not bundled. A regex rule that exceeds 512 matches on a
page fails closed rather than redacting partially.
