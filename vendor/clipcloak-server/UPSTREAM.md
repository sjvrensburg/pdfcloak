## Vendored: ClipCloak server

Upstream: https://github.com/sjvrensburg/clipcloak (commit 5dd8d4f, vendored 2026-10-21).
The browser extension and desktop app were not vendored; ClipCloak is no longer actively
maintained, so this copy is ours to maintain. Keep changes minimal and note them below.

- `src/` — Rust daemon: `POST /classify`, `GET /health`, GLiNER2 via `gliner2-rs`.
- `vendor-gliner2-rs/` — Apache-2.0 inference crate (8 ONNX fragments, `ort =2.0.0-rc.13`).
- ClipCloak code: MIT (see `LICENSE.clipcloak-MIT`). gliner2-rs: Apache-2.0.
- Model weights are NOT vendored: ~620 MB, fetched from Hugging Face on first run
  (`stefanj0/gliner2-sa-names-lora`, `onnx_int8`) or supplied via `PII_MODELS_DIR`.

Local changes (pdfcloak):

- `vendor-gliner2-rs/src/lib.rs` — route `onnx_int8` subfolders to the V2 engine
  (upstream only checked for `"v2"` in the subfolder name).
- `vendor-gliner2-rs/src/lib_v2.rs` — treat `onnx_int8` as fp16-named files for
  download suffix selection (upstream picked `_fp32.onnx`, which 404s; the repo
  ships int8 weights in `*_fp16.onnx` files with no `_iobinding` variants).

  Without these, `clipcloak-server` fails on first run with
  `Failed to download onnx_int8/encoder_fp32.onnx: http status: 404`.
  The `PII_MODELS_DIR` local-dir path was unaffected (engine autodetects V2
  from `token_gather_fp16.onnx`). Routing now matches whole path components.

- `src/lib.rs` — added authenticated `GET /identity` (returns
  `{"service":"clipcloak","ok":true}`) so pdfcloak can prove the listener on
  its spawned port is this sidecar before sending PII; and `/classify` now
  caps the request-body read at `MAX_TEXT_BYTES + 64 KiB` instead of reading
  the whole body before checking the limit.
