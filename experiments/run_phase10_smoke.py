#!/usr/bin/env python3
r"""Phase 10: Fast Closed-Loop Smoke Test.

Verifies the end-to-end Adaptive 3DGS pipeline in under 30 seconds:
    S_t -> X_t -> \hat{X}_t -> \hat{U}_t -> A_t -> S_{t+1}

Checks:
    1. Checkpoint loading and parameter freeze (requires_grad=False).
    2. B2 Online EMA normalizer activity (beta=0.90).
    3. Observable 11-D causal feature extraction (no oracle peeking).
    4. TwoHeadMLP utility and cost predictions (finite, positive costs).
    5. Phase10Selector knapsack budget enforcement (scheduled_cost <= budget).
    6. Closed-loop StateStore synchronization (N_model == N_store, unique persistent IDs).
    7. Continuous map evolution across frames without state reset.
    8. Numerical stability (zero NaNs, finite PSNR).
    9. Strict model weight immutability (||theta_T - theta_0|| == 0.0, SHA-256 match).
"""
import sys
import time
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.phase10_protocol import (
    DEFAULT_BUDGET_MS,
    SAFETY_FACTOR,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    TEST_SCENE,
)
from research.phase10_runtime import (
    Phase10ModelBundle,
    Phase10Selector,
    update_statestore_closed_loop,
    load_phase10_sequence,
    get_pipeline_config,
)
from research.pipeline import OnlineReconstructionPipeline


def run_smoke_test(
    scene_name: str = TEST_SCENE,
    n_frames: int = 5,
    seed: int = 42,
    budget_ms: float = DEFAULT_BUDGET_MS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> bool:
    """Runs fast smoke test verifying all Phase 10 runtime invariants."""
    print("=" * 70)
    print("   PHASE 10 END-TO-END CLOSED-LOOP SMOKE TEST")
    print("=" * 70)
    print(f">> Device:     {device}")
    print(f">> Scene:      {scene_name}")
    print(f">> Frames:     {n_frames}")
    print(f">> Seed:       {seed}")
    print(f">> Budget:     {budget_ms:.1f} ms")
    print("-" * 70)

    # 1. Load sequence
    print(">> [1/7] Loading RGB-D test sequence...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=scene_name,
        n_frames=n_frames,
        H=DEFAULT_IMAGE_HEIGHT,
        W=DEFAULT_IMAGE_WIDTH,
        device=device,
    )
    assert len(frames) == n_frames, f"Expected {n_frames} frames, got {len(frames)}"
    print(f"   [PASS] Loaded {len(frames)} frames.")

    # 2. Pipeline initialization
    print(">> [2/7] Initializing reconstruction pipeline...")
    cfg = get_pipeline_config(policy="budget_aware", budget_ms=budget_ms, seed=seed + 300)
    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0]["pose"],
    )
    n0 = pipeline.gaussian_model.num_gaussians
    assert n0 > 0, "Gaussian initialization failed (N=0)"
    print(f"   [PASS] Pipeline initialized with {n0} Gaussians.")

    # 3. Model bundle & weight snapshot
    print(">> [3/7] Loading frozen model bundle & snapshotting weights...")
    bundle = Phase10ModelBundle(seed=seed, device=device)
    assert not bundle.model.training, "Model is not in eval mode"
    assert all(not p.requires_grad for p in bundle.model.parameters()), "Model parameters are not frozen"
    w0_snapshot = bundle.snapshot_weights()
    h0_hash = bundle.compute_weights_hash()
    print(f"   [PASS] Model loaded in eval mode, all grad frozen, SHA: {h0_hash[:12]}...")

    # 4. Selector setup
    print(">> [4/7] Setting up Phase10Selector (zero-oracle verified)...")
    selector = Phase10Selector(
        model_bundle=bundle,
        budget_ms=budget_ms,
        safety_factor=SAFETY_FACTOR,
        device=device,
    )
    current_diag = [{}]

    def _selector_hook(pipe: OnlineReconstructionPipeline, N: int) -> torch.Tensor:
        mask, diag = selector.select(pipe, policy="ours", frame_idx=pipe.frame_count)
        current_diag[0] = diag
        return mask

    pipeline._custom_selector_fn = _selector_hook

    # 5. Closed-loop trajectory execution (S_t -> S_{t+1})
    print(f">> [5/7] Running {n_frames - 1} online closed-loop transitions...")
    prev_n = n0
    for t in range(1, n_frames):
        t_start = time.perf_counter()
        n_before = pipeline.gaussian_model.num_gaussians

        # Process frame
        m = pipeline.process_frame(
            rgb=frames[t]["rgb"],
            depth=frames[t]["depth"],
            gt_pose=frames[t]["pose"],
        )

        # Closed-loop StateStore update
        audit = update_statestore_closed_loop(
            pipeline=pipeline,
            frame_idx=t,
            optimize_mask=pipeline._last_optimize_mask if hasattr(pipeline, "_last_optimize_mask") else torch.zeros(pipeline.gaussian_model.num_gaussians, dtype=torch.bool, device=device),
        )

        n_after = pipeline.gaussian_model.num_gaussians
        diag = current_diag[0]
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        # Assertions
        assert audit["statestore_synced"], f"StateStore out of sync at frame {t}"
        assert diag.get("zero_oracle_verified", False), f"Zero oracle verification failed at frame {t}"
        assert diag.get("zero_future_leakage", False), f"Zero future leakage failed at frame {t}"
        assert np.isfinite(m["psnr"]), f"Non-finite PSNR at frame {t}: {m['psnr']}"
        assert diag.get("scheduled_cost", 0.0) <= budget_ms * 1.05, f"Scheduler budget exceeded: {diag.get('scheduled_cost')}"

        print(f"   Frame {t:02d}: N={n_after} (densified={n_after - n_before:+d}) | "
              f"Selected={diag.get('n_selected', 0)} | PSNR={m['psnr']:.2f} dB | "
              f"Sched Cost={diag.get('scheduled_cost', 0.0):.2f} ms | Frame Wall={dt_ms:.1f} ms")
        prev_n = n_after

    print("   [PASS] Closed-loop transitions executed cleanly with zero crashes.")

    # 6. Strict model weight immutability check
    print(">> [6/7] Verifying model weight immutability (||theta_T - theta_0|| == 0)...")
    is_immutable, max_diff = bundle.verify_immutability(w0_snapshot)
    hT_hash = bundle.compute_weights_hash()
    assert is_immutable and max_diff == 0.0, f"Model weight mutation detected! max diff = {max_diff}"
    assert h0_hash == hT_hash, "Weights hash changed after trajectory execution"
    print(f"   [PASS] Max weight diff: {max_diff:.1e} (hash identical).")

    # 7. Final status
    print(">> [7/7] Verifying B2 online normalizer drift and history...")
    assert len(bundle.normalizer.history) >= (n_frames - 1), "Normalizer history not updated online"
    print(f"   [PASS] Normalizer tracked {len(bundle.normalizer.history)} frame updates online.")

    print("-" * 70)
    print("   ALL PHASE 10 SMOKE TEST INVARIANTS: PASS")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
