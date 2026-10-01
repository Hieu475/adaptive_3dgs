# CUDA prototypes — RESEARCH ONLY, NOT production path

Production rasterization routes strictly through `gsplat v1.5.3`
(see `research/phase10_runtime.py`, Phase 13 report).

Files in this directory (`rasterize.cu`, `radix_sort.cu`, `binning.cu`,
`preprocess.cu`, `depth_render.cu`, `statistics.cu`) are isolated research
prototypes. They are NOT invoked during frozen benchmarks
(`results/phase13_frozen_benchmark/`, `results/final_confirmation/`).

Do NOT cite their micro-benchmarks as end-to-end FPS.
To productize: fuse attribution tracing (`render_with_attribution`,
currently ~0.85ms fast-approx vs 500ms naive) into a native gsplat kernel,
then re-run `experiments/run_phase13_frozen_benchmark.py` + manifest verify.

Status: KEPT for reference, EXCLUDED from `setup.py` production build.
