# pdfcloak

Remove personal information from PDFs before you share them.

pdfcloak finds names, email addresses, phone numbers, ID numbers, and other
details you define. It then deletes that text from the file.
It does not draw black boxes over the text.
Covered text can still be copied out of many PDFs; deleted text cannot.

Everything runs on your machine. No PDF is uploaded anywhere.
Detection uses your own regex rules, an on-device AI model, or both.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) (recommended) or
  [pipx](https://pipx.pypa.io/).
- Optional, only for AI detection: a Rust toolchain
  ([rustup.rs](https://rustup.rs)) and `onnxruntime` (see below).

Regex-only use needs no AI model, no Rust, and no extra downloads.

## Install from GitHub

pdfcloak is not on PyPI yet. Install it straight from this repository.

Recommended, with uv:

```bash
uv tool install git+https://github.com/sjvrensburg/pdfcloak
pdfcloak --version
```

With pipx:

```bash
pipx install git+https://github.com/sjvrensburg/pdfcloak.git
pdfcloak --version
```

With pip, inside a virtual environment:

```bash
pip install git+https://github.com/sjvrensburg/pdfcloak.git
```

To pin an exact version instead of taking the latest `main`, add
`@<branch-or-commit>` to the URL. For example:

```bash
uv tool install git+https://github.com/sjvrensburg/pdfcloak@main
```

Upgrade later with `uv tool upgrade pdfcloak` or
`pipx upgrade pdfcloak`. Remove it with `uv tool uninstall pdfcloak`
or `pipx uninstall pdfcloak`.

### Try it without installing

```bash
uvx --from git+https://github.com/sjvrensburg/pdfcloak pdfcloak scan in.pdf
```

## Quick start

Check a file first. This never changes the PDF:

```bash
pdfcloak scan in.pdf
```

Redact it. Always write to a new file:

```bash
pdfcloak redact in.pdf -o out.pdf --regex email:'[A-Za-z0-9._%+-]+@[^ ]+'
```

Check the result:

```bash
pdfcloak verify out.pdf
```

`redact` checks its own output before it finishes.
Run `verify` yourself when you want a second, independent pass.

## Common tasks

### Use your own patterns

Repeat `--regex name:pattern` for each kind of detail:

```bash
pdfcloak redact in.pdf -o out.pdf \
  --regex email:'[A-Za-z0-9._%+-]+@[^ ]+' \
  --regex phone:'\+?[0-9][0-9 ]{7,}'
```

Or keep the rules in a TOML file and reuse them:

```toml
[[rule]]
name = "email"
pattern = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"

[[rule]]
name = "sa_id"
pattern = "\\b\\d{13}\\b"
```

```bash
pdfcloak redact in.pdf -o out.pdf --rules rules.toml
```

Regex matches take priority over AI matches.
Your `--regex`/`--rules` rules are *added to* the built-in email/url/ipv4
rules — they never silently replace them. Pass `--no-builtins` if you really
only want your own rules. pdfcloak prints the active rule list when you pass
custom rules.

Reports never print matched text. Each match carries a masked glimpse
(`j***@e***.com`) so you can triage without leaking the value. Set
`PDFCLOAK_REPORT_HMAC_KEY` if you need a keyed digest for audit matching.
Add `--show-matches` only if you really need to see the matched text.

### Limit the pages

Pages are numbered from 1:

```bash
pdfcloak scan in.pdf --pages 1,3-5,last
pdfcloak redact in.pdf -o out.pdf --pages 3- --rules rules.toml
```

Useful forms: `all`, `3`, `3-7`, `3-`, `-7`, `1,4,last`.

### Other options

- `--replacement label` writes a label such as `[EMAIL]` instead of a
  black bar. The default is `--replacement black`. Labels are sized to fit
  the text they replace.
- `--password <password>` opens an encrypted PDF. Prefer
  `--password-env VAR` or the `PDFCLOAK_PASSWORD` environment variable:
  a password on the command line is visible in shell history and process
  listings. On a terminal, pdfcloak prompts for it instead. Note that the
  redacted output is *not* re-encrypted.
- `--report report.json` saves the JSON report to a file (written
  atomically, owner-only). Reports include counts by label and page;
  `--limit`/`--offset` page through large reports.
- `--dry-run` (on `redact`) shows what would be removed without
  writing a file, including a `scrub_preview` of the hidden carriers
  (metadata, bookmarks, attachments, annotations, form fields, links)
  that scrubbing would delete.
- `--preview preview.pdf` (on `redact`) writes an annotated copy with the
  matches highlighted, so you can inspect before anything is deleted.
- `--exclude-labels url` (on `redact`) leaves matches with those labels
  in place. `--interactive` asks for confirmation on a terminal first.
- `--fail-on-findings` (on `scan`) exits 3 when matches — or pages that
  could not be checked — are found.
- `--quiet` silences progress and summaries (stdout JSON is unaffected);
  `--verbose` lists the detectors and active rules.
- `--max-pages N` / `--max-mb N` refuse oversized inputs instead of
  running out of memory on them.

While it works, pdfcloak prints page progress and a one-line summary
(`found: 3 match(es) (email=2, sa_id=1)`) to stderr, plus the suggested
next command. Existing outputs are replaced atomically — a crash never
leaves a half-written file — and a file that fails verification is moved
to `<output>.unsafe` instead of sitting at the path you meant to share.

`redact` also strips hidden carriers of personal data on every page:
document metadata, attachments, annotations, form fields, link targets,
and bookmarks. Text on pages outside `--pages` is left as it is.

### Scanned PDFs, folders, and a local web page

- `pdfcloak ocr in.pdf -o ocr.pdf` adds a text layer to a scanned PDF
  (needs [ocrmypdf](https://ocrmypdf.readthedocs.io) on PATH; pdfcloak
  itself has no OCR engine). Redact the OCR'd file afterwards.
- `pdfcloak batch scan <dir>` / `pdfcloak batch redact <dir> -o <outdir>`
  processes every PDF in a folder and prints one JSON summary.
- `--profile gdpr|hipaa|legal|sa|minimal` bundles labels, extra rules, and
  (for hipaa) a threshold for a policy context. It is a starting point,
  not a compliance guarantee; your own rules and `--labels` still merge in.
- `--lang af|de` translates the human verdict lines (the JSON report is
  unchanged). Page progress and detector names stay as they are.
- `pdfcloak serve` starts a small web page at http://127.0.0.1:8765 for
  scanning or redacting one PDF in the browser. It is loopback-only and
  has no authentication — do not expose it to a network.

Reading order honours right-to-left and vertical text: RTL lines are
normalised to logical order, vertical columns read top-to-bottom,
right-to-left, and redaction boxes merge in either writing direction.

## AI detection (optional)

Regex only finds what you can describe as a pattern.
The AI detector (GLiNER2) also finds names and other details in context.
It runs on your device; your PDF still never leaves the machine.

One-time setup. Install (or reinstall) with `onnxruntime` included:

```bash
uv tool install --with onnxruntime git+https://github.com/sjvrensburg/pdfcloak
# already installed with pipx? use: pipx inject pdfcloak onnxruntime
pdfcloak sidecar build     # needs Rust; builds the detector binary
pdfcloak sidecar status    # check that everything is in place
pdfcloak sidecar doctor    # preflight: binary, onnxruntime, model cache,
                           # with the exact fix for anything missing
```

`sidecar doctor` exits non-zero when the AI detector is not ready, so it
can gate a setup script. (`sidecar status` stays informational and always
exits 0.)

Then add `--detector clipcloak` to any command:

```bash
pdfcloak redact in.pdf -o out.pdf --detector clipcloak --rules rules.toml
```

The first AI run downloads about 412 MB of model weights from
Hugging Face. Later runs reuse them.
If the sidecar cannot find `libonnxruntime.so`, set `ORT_DYLIB_PATH`
to its path. Harmless `libonnxruntime_providers_openvino.so` warnings
in the sidecar log just mean it fell back to CPU.

## Commands

- `scan` — report what would be found. Never modifies the PDF.
- `redact` — remove the matches, clean hidden data, save, and verify.
- `verify` — re-check a redacted PDF.
- `ocr` — add a text layer to a scanned PDF (needs ocrmypdf).
- `batch scan|redact` — process every PDF in a directory.
- `serve` — local web page for one-off scans/redactions.
- `sidecar build|status|doctor` — build, inspect, or preflight the AI detector.

Exit codes tell scripts what happened:

- `0` — clean, or redaction finished.
- `2` — something is wrong with the input or the command.
  You get a short error message, never a traceback.
- `3` — personal data was found, survived redaction, or could not
  be checked. Treat the file as not safe to share yet.

## Limits

- No built-in OCR. Scanned, image-only pages are flagged in the report;
  use `pdfcloak ocr` (ocrmypdf) to add a text layer first.
  `verify` and `redact` exit `3` for them so you do not share a file
  that was never checked. `--allow-scanned` accepts that risk.
- Matches cannot span a page break.
- Model weights are not bundled; see the download note above.
- A single rule matching more than 512 times on one page stops with
  an error instead of redacting only part of the page.

## For developers

```bash
git clone https://github.com/sjvrensburg/pdfcloak.git
cd pdfcloak
uv sync --group dev
uv run pytest
uv run pdfcloak scan in.pdf
```

Background reading: [pdf-editor-redaction-report.md](pdf-editor-redaction-report.md)
explains why text is deleted rather than covered, and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) credits the
reading-order code ported from
[RailReaderCore](https://github.com/sjvrensburg/RailReaderCore) and the
vendored [ClipCloak](vendor/clipcloak-server/UPSTREAM.md) sidecar.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).
