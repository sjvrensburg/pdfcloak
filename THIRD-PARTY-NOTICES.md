# Third-party notices

## RailReaderCore (MIT)

The reading-order code in `src/pdfcloak/extract.py` (Docstrum block
grouping, XY-Cut++ validated/projection column gutters, divider handling,
margin-note masking) is a Python port of parts of
[RailReaderCore](https://github.com/sjvrensburg/RailReaderCore)
(`TextLayoutAnalyzer` and `XYCutPlusPlusResolver`), which in turn
implements Liu, Li & Wei (2025), "XY-Cut++: Advanced Layout Ordering via
Hierarchical Mask Mechanism" (arXiv:2504.10258).

MIT License — Copyright (c) 2024-2026 Stefan van Rensburg. The full MIT
text ships with RailReaderCore; this notice satisfies attribution for the
ported constants and algorithms. Adaptations: operates on PyMuPDF lines
instead of layout-model blocks, no role/heading logic (pdfcloak has no
block classes), content-stream index used only as a within-row tie-break.

## ClipCloak / gliner2-rs

See `vendor/clipcloak-server/UPSTREAM.md` (MIT / Apache-2.0).
