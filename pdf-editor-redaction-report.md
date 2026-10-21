# Open-Source PDF Editors and a CLI PII Redactor: Deep Dive

**Build the CLI on PyMuPDF, not on an editor. Reuse ClipCloak as a pluggable detector, not as the PDF engine.**

That is the single decision everything else follows from. No open-source editor edits PDF text the way a word processor edits a document, and only one mainstream open library gives you a true, physical redaction primitive from Python.

Nothing was implemented. This report was the deliverable of the research task.

## 1. What open-source PDF editors actually do

There are four patterns. Only the fourth can redact.

### Pattern A — Import to native objects, re-export

- **LibreOffice Draw** — Uses its `pdfimport` filter (historically Xpdf-derived) to convert each page into Draw shapes and text objects. You edit the shapes, then LibreOffice re-exports a newly generated PDF. Layout and fonts often shift. Multi-page handling is weak.
- **Inkscape** — Uses an internal Poppler-derived importer, or external Poppler with Cairo. PDF content becomes SVG text or glyph paths. Good for one-page graphics, poor for documents.
- **Scribus** — A DTP application using PoDoFo and Poppler-related PDF handling. PDFs are imported or placed into frames, then a new document is exported.

These tools do not edit the original content stream. They translate it, let you edit the translation, then generate a different PDF.

### Pattern B — Page composition only

- **PDF Arranger** — A Python/GTK frontend for [pikepdf](https://github.com/pikepdf/pikepdf), which binds QPDF. It merges, splits, rotates, crops and reorders pages. It cannot safely edit text in place.
- **PDFsam Basic** — Uses SAMBox, a PDFBox fork. Again: page operations, not text editing.
- **Sejda** — Also built on the SAMBox/PDFBox lineage.
- **Stirling-PDF** — A Spring Boot application. Its current build files specify [Apache PDFBox 3.0.8](https://github.com/Stirling-Tools/Stirling-PDF/blob/main/app/common/build.gradle) plus its `jpdfium` PDFium binding. It offers many operations, including redaction workflows, but it is a server platform built on libraries, not a new PDF engine.

This pattern is useful for selecting or extracting pages before redaction. It is not sufficient for redaction itself.

### Pattern C — Annotation overlay only

- **Okular and Evince** — Poppler-based viewers. They can add annotations, highlights and form entries.
- **Xournal++** — Renders the PDF as a background using Poppler. Your handwriting and text sit in a separate layer. Export uses Cairo.
- **Firefox / PDF.js** — Adds FreeText, ink and stamp annotations. It does not rewrite existing page text.

This is the dangerous pattern. Drawing a black rectangle over text is annotation, not redaction. The original characters remain selectable, searchable and extractable. Many public redaction failures have exactly this cause.

### Pattern D — Programmatic content-stream manipulation

This is where a CLI belongs. A PDF page is drawn by operators in a content stream: `BT … Tj/TJ … ET`. The operands are usually font-specific glyph or character codes, not Unicode strings. Fonts may use custom encodings, ligatures, kerning arrays and optional `/ToUnicode` maps. There is no paragraph, line or word object stored in the file.

So "edit this text" really means:

1. Decode glyphs to Unicode.
2. Infer characters, words, lines and reading order from their coordinates.
3. Locate rectangles for the target characters.
4. Rewrite the content stream so those glyphs are never drawn.
5. Remove or blank affected image pixels and vector graphics.
6. Save without incremental updates, then garbage-collect unreferenced objects.

## 2. Library shortlist

| Library | Stack / licence | Redaction ability | Verdict |
|---|---|---|---|
| PyMuPDF | Python / C, AGPL or commercial | Native, character-level | Recommended core |
| pikepdf / QPDF | Python / C++, MPL / Apache | Manual stream editing | Cleanup and fallback |
| pypdf + pdfminer.six | Python, BSD / MIT | Extraction, no redaction | Too low-level alone |
| Apache PDFBox | Java, Apache-2.0 | Manual stream editing | Good if you choose JVM |
| Poppler / PDFium | C++, GPL-ish / BSD-like | Rendering, extraction | Not a redaction core |
| lopdf / pdfcpu | Rust / Go, MIT / Apache | Manual stream editing | Too much PDF work for MVP |

### Why PyMuPDF wins

[PyMuPDF](https://github.com/pymupdf/PyMuPDF) binds MuPDF, a C engine. Per page it gives you:

- `get_text("dict" | "rawdict")` — blocks, lines, spans and individual character bounding boxes.
- `search_for(text)` — ready-made rectangles for exact strings.
- `add_redact_annot(rect, text=…, fill=…)` — queue a redaction rectangle.
- `apply_redactions(images=…, graphics=…, text=…)` — apply all queued redactions.

The [PyMuPDF redaction documentation](https://pymupdf.readthedocs.io/en/latest/page.html) is explicit: with the default `PDF_REDACT_TEXT_REMOVE`, every character whose box overlaps a redaction rectangle is "physically removed from the page" and "will no longer appear in text extractions." Images can be pixel-blanked (`PDF_REDACT_IMAGE_PIXELS`, the default) or removed. Covered vector graphics can be removed too. Redaction annotations are consumed by applying them, so no live annotation is left behind.

No permissive alternative offers this as a finished primitive. With pikepdf, pypdf or lopdf you would have to parse `Tj`/`TJ` operands, resolve font encodings and `ToUnicode` CMaps, and rewrite streams without breaking kerning or shared resources. That is a research project, not an MVP.

### The licence gate

MuPDF and PyMuPDF are dual-licensed: AGPL-3.0, or a commercial licence from Artifex. This is the one real cost of the recommendation.

- If the tool is open-sourced under AGPL-compatible terms, proceed with PyMuPDF.
- If it must be closed or permissively licensed, either buy the commercial licence or accept the much larger pikepdf-based build.
- Do not defer this decision. It determines the core dependency.

## 3. What ClipCloak already solved

I inspected [sjvrensburg/clipcloak](https://github.com/sjvrensburg/clipcloak) directly. It is more reusable for this tool than a generic "export a model to ONNX" plan.

### Architecture

- A Rust daemon using `tiny_http` binds only `127.0.0.1:8731`.
- `POST /classify {text, threshold?}` returns `{entities: [{label, text, start, end, score}], redacted, parts}`.
- `GET /health` reports readiness.
- A bearer token (`PII_TOKEN`) protects the endpoint. CORS is pinned to the extension origin.
- The browser extension is a thin client. The Tauri tray app wraps the same server library for end users.

### Model

- GLiNER2, 205M parameters, schema-driven: you pass the entity labels at inference time.
- Default model: `stefanj0/gliner2-sa-names-lora`, `onnx_int8` subfolder, fetched from Hugging Face on first run. `PII_MODELS_DIR` points at local files to skip the download. `PII_MODEL_REPO` and `PII_SUBFOLDER` override the source.
- Critically, GLiNER2 **cannot be a single ONNX graph**. It is eight fragments — encoder, `schema_gather`, `count_*`, `token_gather`, `span_rep`, scorer, classifier — orchestrated by the vendored `gliner2-rs` crate, plus `tokenizer.json`. About 620 MB in fp16.
- Inference uses the Rust `ort` crate pinned to `=2.0.0-rc.13`, with `load-dynamic`. `ORT_DYLIB_PATH` must point at `libonnxruntime.so`. That pin is load-bearing; the repo documents several breaking API changes across `ort` release candidates.
- GPU acceleration via a WebGPU execution-provider plugin is optional. CPU fallback is the default path.

### Detection lessons worth copying

- Default labels are `name`, `street address`, `email`, `phone_num`, `id_num`, `url`, `username`. Label wording matters measurably: `street address` scores 0.84–0.98 on spans where bare `address` scores 0.44–0.51. Default threshold is `0.55`.
- Long text is windowed: 1,500-byte windows, 300-byte overlap, offsets remapped, entities pooled, masked once. This exists because inference is roughly O(n²): 5,000 characters took about 37 seconds in one pass.
- User regex rules are already implemented the right way. A rule match gets `RULE_SCORE = 2.0`, higher than any model score, so **rules always win overlaps**. Rules are size-limited (1 MiB compiled program), capped at 512 matches per rule, and empty matches are skipped. Rule names become mask labels, so `Account number` masks to `[ACCOUNT_NUMBER]`.

In other words, ClipCloak already built the detector service, the regex layer, the overlap resolution and the windowing. The PDF tool should consume it, not reimplement it.

### Three ways to hook the model in

**Option 1 — Call the ClipCloak daemon (recommended for v1).** The CLI starts or connects to the daemon and posts page text to `/classify`. Zero model code in the PDF tool. The daemon stays a separate Rust binary, exactly as it ships today. Downside: users must run two processes, and the token must be managed.

**Option 2 — Spawn the daemon as a sidecar.** Same protocol, but the CLI launches `clipcloak-server` itself with `PII_MODELS_DIR`, a chosen port and a generated token, then kills it on exit. This is how the Tauri app already treats the server library. Best user experience of the three, at the cost of packaging a Rust binary alongside a Python CLI.

**Option 3 — In-process ONNX in Python.** Use `onnxruntime` plus `tokenizers`. This is straightforward only for single-graph token-classification models — for example a Piiranha-style or ai4privacy-style DistilBERT, exported with Optimum, with `word_ids()` and offset mappings converting sub-word predictions back to character spans. It is **not** straightforward for GLiNER2: you would have to re-orchestrate eight ONNX sessions and the schema/scorer logic that `gliner2-rs` already implements. Do not port that to Python. If you want GLiNER2, use Option 1 or 2.

A fourth plugin is cheap to add later: Microsoft Presidio's analyzer as a detector. It gives you mature regex recognisers with validators (Luhn for cards, checksums for some national IDs), but adds spaCy as a heavy dependency. Treat it as an optional recogniser pack, not the core.

## 4. Recommended design

Working name: `pdfcloak`, to sit beside ClipCloak.

### Pipeline

1. **Open and scope.** Parse the page specification first. Open with PyMuPDF and validate the range against the page count before doing any expensive work.
2. **Extract with geometry.** For each selected page, call `get_text("rawdict")`. Build one page string plus a map from character offset → character rectangle. Preserve span boundaries; insert spaces or newlines between blocks and lines deliberately, and record them in the offset map as characters with no rectangle.
3. **Detect.** Run regex rules first, then the model over windowed page text. Both return the same span type: `{page, start, end, label, score, source}`. Never send text from unselected pages to the model.
4. **Resolve overlaps.** Pool all spans per page. Sort by score descending, then longest span, then earliest start — mirroring ClipCloak's `dedup_entities`. Regex spans, scored above model spans, win.
5. **Map spans to rectangles.** Convert each surviving span into one rectangle per line fragment. A name split across two lines becomes two rectangles, not one tall box covering unrelated text. Expand each rectangle by roughly half a point to catch glyph overhang, but take care: PyMuPDF removes any character whose box merely *overlaps* a redaction rectangle, so aggressive expansion deletes neighbours.
6. **Queue and apply.** `page.add_redact_annot(rect, fill=(0,0,0))` for each rectangle, then one `page.apply_redactions()` per page. Optionally pass replacement text such as `[EMAIL]` to `add_redact_annot`; be aware replacement text uses an inserted textbox, may not fit the original space, and its font will differ. Black bars are the safer default, with `[LABEL]` as an opt-in.
7. **Scrub the rest of the file.** Redaction rectangles only cover page content. Also clear document metadata and XMP, remove attachments and embedded files, delete remaining annotations and form-field values that contain PII, and remove links overlapping redacted text (PyMuPDF does this last one for you).
8. **Save safely.** Never save incrementally — the original bytes would survive in the file. Use `doc.save(out, garbage=4, deflate=True, clean=True)`. `garbage=4` removes unreferenced objects left behind by deleted content; `clean=True` sanitises content streams.
9. **Verify.** Reopen the output. Re-extract text from the selected pages and re-run every detector. Fail loudly if any hit survives. Also assert the PII strings do not appear anywhere in the raw output bytes. Report per-page counts, but do not print the matched PII into logs or reports by default; offer an explicit `--show-matches` flag for local debugging.

### Proposed CLI surface

```bash
pdfcloak redact in.pdf -o out.pdf \
  --pages 1,3-5,last \
  --rules rules.toml \
  --detector clipcloak \
  --labels name,email,phone_num,id_num,"street address" \
  --threshold 0.55

pdfcloak redact in.pdf --pages 2-4 --regex email:'[A-Za-z0-9._%+-]+@…' --dry-run
pdfcloak scan in.pdf --pages all --detector regex --report report.json
```

Three subcommands are worth separating:

- `scan` — detect only, write a JSON report. Never modifies the PDF.
- `redact` — detect, apply, scrub, save, verify.
- `verify` — re-check an already-redacted PDF. This is what a cautious user runs before sharing a file.

`--dry-run` on `redact` should mean "scan plus show the rectangles that would be applied", ideally by rendering affected pages to PNG for eyeball review.

### Page range grammar

Keep it small, 1-based and inclusive, matching how users cite pages:

| Spec | Meaning |
|---|---|
| `all` | Every page |
| `3` | Page 3 only |
| `3-7` | Pages 3 to 7 inclusive |
| `3-` | Page 3 to the end |
| `-7` | Page 1 to page 7 |
| `1,4,9-11,last` | Union of the above |
| `first`, `last` | Named endpoints |

Internally convert to 0-based indices immediately and store a `set[int]`. Reject reversed ranges (`7-3`), out-of-range pages and empty selections with a clear error. Do not support negative Python-style indices; `-1` is ambiguous next to range syntax.

### Regex rules file

Reuse ClipCloak's semantics: a named rule, an enabled flag, and a pattern. The name becomes the label.

```toml
[[rule]]
name = "email"
pattern = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"

[[rule]]
name = "sa_id"
pattern = "\\b\\d{6}[0-9]{4}[0-9]{3}\\b"
enabled = true
```

Design points:

- Named rules beat bare patterns because reports and replacement labels need names. Still allow `--regex name:pattern` on the command line for one-offs.
- Validate all patterns at startup. Report which rule failed and why, then exit. Do not silently skip a bad rule — a silently skipped rule looks identical to "no PII found."
- Python's `re` can catastrophically backtrack, unlike Rust's linear-time `regex` crate used in ClipCloak. Prefer the third-party `regex` module, which supports timeouts, and set one (for example 1 second per rule per page). Also cap matches per rule per page, as ClipCloak does at 512.
- Offer checksum validation as a rule option later (`validate = "luhn"`). Regex alone produces many false positives on card-like and ID-like numbers.

### Detector interface

One protocol, three implementations:

```python
class Detector(Protocol):
    name: str
    def detect(self, text: str, *, labels: list[str],
               threshold: float) -> list[Span]: ...
```

- `RegexDetector` — compiled rules from TOML and `--regex`.
- `ClipCloakDetector` — HTTP client for `/classify`. Handles the token, health check, windowing and offset remapping.
- `OnnxDetector` — `onnxruntime` + `tokenizers` for a single-graph token-classification model. Includes a `--model` path (or Hugging Face ID), BIO/BIOES decoding, and sub-word-to-character offset conversion.

Every detector returns character offsets into the page string. None of them knows anything about PDFs. That separation is what lets you test detection on plain text fixtures without generating PDFs.

## 5. Hard parts and failure modes

**Offset mapping is the core risk.** Regex and the model run on a reconstructed page string. That string is your invention: block order, inserted spaces, hyphenated line breaks (`informa-\ntion`) and ligature characters (`ﬁ`) all affect what matches and where it lands. Build the offset map as you build the string, and write dedicated tests for multi-span, multi-line and hyphenated matches. Fall back to `page.search_for(matched_text)` when offset mapping fails, and log a warning when neither works — never silently drop a detection.

**Matches must not cross pages.** Detect per page. A phone number split across a page break will be missed; accept that, and let users add a rule or page range rather than engineering cross-page joins.

**Scanned PDFs have no text layer.** `get_text` returns nothing. Detect this per page (near-zero characters but a large image) and route those pages through OCR: render at 300 DPI, OCR with Tesseract or RapidOCR, and redact the resulting word boxes as rectangles. Note the asymmetry: for scanned pages the PII also lives in the image pixels, so `PDF_REDACT_IMAGE_PIXELS` matters, and OCR box errors mean you should pad rectangles more generously. Make this `--ocr auto|never|always`, defaulting to `auto` with a printed warning listing which pages were OCR'd.

**PII hides outside page text.** Metadata, XMP, attachments, annotations, AcroForm values, bookmarks and file names can all carry names. Step 7 of the pipeline is not optional polish.

**Partial-glyph deletion looks broken.** Because PyMuPDF deletes whole characters on any overlap, redacting part of a word leaves the rest. That is correct behaviour for privacy. Do not try to be clever about it.

**Length leakage.** A black bar exactly the width of `john.smith@example.com` reveals its length. ClipCloak's `[LABEL]` masking has the same property in text form. Offer a fixed-width bar option for high-sensitivity use, and mention the trade-off in `--help`.

**Performance.** Model inference dominates. Window per page, batch nothing across pages, cache the daemon connection, and print progress per page. A 200-page document with the model on every page will be slow; page scoping is also your performance feature.

**False positives destroy documents.** Redaction is irreversible. `scan` first, `--dry-run` second, `redact` third. Also write the report to a file by default so a user can audit what was removed, and consider a `--min-score` per label.

## 6. Suggested phasing

1. **v0.1** — PyMuPDF core, page-range parser, `RegexDetector`, `scan`/`redact`/`verify`, metadata scrub, safe save, self-verification. No model. This is already a useful tool.
2. **v0.2** — `ClipCloakDetector` over HTTP (Option 1). Reuses your existing daemon, labels, threshold and windowing unchanged.
3. **v0.3** — Sidecar mode (Option 2) so users get one command. OCR routing for scanned pages.
4. **v0.4** — `OnnxDetector` for a single-graph model, checksum validators, fixed-width bars, batch directory processing.

Test against a small corpus you generate yourself: born-digital PDFs from ReportLab with known PII at known positions, a hyphenated document, a scanned page, a form with PII in field values, and a document with PII in metadata. The decisive test is always "extract the output and assert the string is absent", not "the page looks black."

## Bottom line

Do not build an editor, and do not draw boxes. Build a thin, well-tested pipeline: **PyMuPDF for geometry and physical removal, your existing ClipCloak daemon for AI detection, regex as the highest-priority detector, and a mandatory verify pass before you call anything redacted.**

The two decisions to make before writing code are the licence question (AGPL versus an Artifex commercial licence) and Option 1 versus Option 2 for shipping the model.
