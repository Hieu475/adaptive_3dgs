# DEPRECATED — pose-bugged substrate, do NOT cite absolutes

This `final_confirmation/` snapshot (2026-09-19, OURS 19.24 dB) was generated
BEFORE the pose-convention fix.

Root cause: TUM `groundtruth.txt` stores Camera-to-World (C2W), but
`research/projection.py` requires World-to-Camera (W2C). The old
`research/phase10_runtime.py::load_phase10_sequence` passed `item["pose"]`
straight through (C2W → renderer), so every frame rendered from a wrong
viewpoint and PSNR was artificially depressed.

The current loader applies `torch.inverse()` (C2W→W2C) plus depth hygiene
(inpaint + Sobel edge removal + 0.4–4m range) and stricter densification gating
(`transmission>=0.20`, throttle only when `error_ratio<0.15`).

Policy DELTAS remain directionally valid (OURS +2.87 dB vs error_influence),
but ABSOLUTE PSNR values here are NOT comparable to post-fix runs
(e.g. `results/rtg_slam_150f_8seeds/` OURS 27.67 dB).

Kept for provenance only. Authoritative corrected-substrate benchmark:
`results/phase14_corrected/` (in progress).
Manifest JSON in this directory is FROZEN — do not modify.
