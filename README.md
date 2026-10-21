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
Reports list a short hash of each match, not the text itself.
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
  black bar. The default is `--replacement black`.
- `--password <password>` opens an encrypted PDF.
- `--report report.json` saves the JSON report to a file.
- `--dry-run` (on `redact`) shows what would be removed without
  writing a file.

`redact` also strips hidden carriers of personal data on every page:
document metadata, attachments, annotations, form fields, link targets,
and bookmarks. Text on pages outside `--pages` is left as it is.

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
```

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
- `sidecar build|status` — build and inspect the optional AI detector.

Exit codes tell scripts what happened:

- `0` — clean, or redaction finished.
- `2` — something is wrong with the input or the command.
  You get a short error message, never a traceback.
- `3` — personal data was found, survived redaction, or could not
  be checked. Treat the file as not safe to share yet.

## Limits

- No OCR yet. Scanned, image-only pages are flagged in the report.
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
