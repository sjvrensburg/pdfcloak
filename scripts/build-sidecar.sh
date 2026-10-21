#!/usr/bin/env bash
# Build the vendored ClipCloak sidecar (Option 2 distribution) and stage the
# binary where pdfcloak looks for it: vendor/clipcloak-server/target/release/
# Requires: cargo (Rust) and a libonnxruntime.so at runtime (load-dynamic) —
# install the onnx extra (`uv pip install 'pdfcloak[onnx]'`) so
# pdfcloak.sidecar.find_onnxruntime_lib() can locate it, or set ORT_DYLIB_PATH.
set -euo pipefail
cd "$(dirname "$0")/../vendor/clipcloak-server"
echo "Building clipcloak-server (release)…"
cargo build --release --locked
ls -lh target/release/clipcloak-server
echo "Done. First run downloads ~412 MB of GLiNER2 int8 weights from Hugging Face"
echo "unless PII_MODELS_DIR / --models-dir points at local ONNX fragments."
