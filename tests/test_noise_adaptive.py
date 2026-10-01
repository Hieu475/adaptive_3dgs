import numpy as np
from research.noise_adaptive import (
    frame_noise_score, frame_noise_score_v1, frame_noise_score_v2,
    frame_noise_score_v3, NoiseAdaptiveController,
)
from research.holdout_eval import compute_ate, summarize_holdout
import torch


def test_clean_scores_lower_than_noisy():
    rng = np.random.default_rng(0)
    clean = np.full((60, 80), 2.0, dtype=np.float32)  # perfect synthetic depth
    noisy = np.where(rng.random((60, 80)) < 0.4, 0.0, 2.0)  # 40% holes
    assert frame_noise_score(clean)["noise"] < frame_noise_score(noisy)["noise"]


def test_controller_routes_extremes():
    c = NoiseAdaptiveController()
    p_clean, k_clean = c.route(0.0)
    p_noisy, k_noisy = c.route(1.0)
    assert p_clean == "error_only" and k_clean >= 0.95
    assert p_noisy == "error_influence_temporal" and k_noisy <= 0.90


def test_v3_separates_holes_from_far_geometry():
    # v3 uses missing-measurement rate (sensor physics), not invalid range
    # (which conflates far-but-valid geometry, e.g. replica_room0's 18% >4m).
    replica = frame_noise_score_v3(raw_holes=0.0005, raw_edge=0.015)
    tum = frame_noise_score_v3(raw_holes=0.25, raw_edge=0.065)
    assert replica["noise"] < 0.10
    assert tum["noise"] > 0.30
    c = NoiseAdaptiveController()
    assert c.route(replica["noise"])[0] == "error_only"
    assert c.route(tum["noise"])[0] == "error_influence_temporal"


def test_ate_zero_on_identical_trajectories():
    poses = [torch.eye(4) for _ in range(5)]
    out = compute_ate(poses, poses)
    assert out["ate_rmse_m"] == 0.0 and out["n_poses"] == 5


def test_holdout_summary_gap():
    s = summarize_holdout([30.0, 30.0], [28.0, 28.0])
    assert s["gen_gap_db"] == 2.0 and s["n_holdout"] == 2
