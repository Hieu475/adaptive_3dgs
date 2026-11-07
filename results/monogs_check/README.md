Integration check with MonoGS (https://github.com/muskie82/MonoGS, commit at clone time, run in a separate conda env).
Local patches (outside this repo): numpy-2 `dtype=str`, evo>=1.3 `align`, env overrides MONOGS_BETAS / MONOGS_LR_MULT / MONOGS_MAP_ITERS /
MONOGS_MAX_FRAMES / MONOGS_REFINE_ITERS, `#include <cstdint>` / `<cfloat>` for GCC 13 builds. One run per setting; run-to-run noise
was about 0.15 dB PSNR and 0.1 cm ATE between two identical default runs (18.76/1.50 cm and 18.90/1.57 cm).
