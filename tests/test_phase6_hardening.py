"""Comprehensive Unit Tests for Phase 6 Scientific Hardening and Unification.

Covers VIỆC 1 through VIỆC 12:
  1. test_group_identity_is_exact: g = (scene, frame, sorted(context_ids))
  2. test_grouped_batch_sampler_no_split: guarantees candidate groups are never split across batches
  3. test_phase6_loss_requires_group_ids_and_tie_filtering: enforces group_ids and filters |ΔU| <= tau
  4. test_listwise_kl_loss_numerical: KL ≈ 0 for matching rankings, KL > 0 for inverted rankings
  5. test_residual_target_invariant: r* = U*(i|S) - U*(i|∅) and r*(∅) = 0
  6. test_empty_context_identity_p4_vs_p6: |U_hat_P6(i|∅) - U_hat_P4(i)| < 1e-6
  7. test_p4_freeze_and_sha256_invariant: P4 backbone SHA256 hashes immutable during training
  8. test_checkpoint_schema_phase6_v2: schema_version "phase6-v2" validation
  9. test_frozen_context_predictor_requires_p4: fails loudly if P4 checkpoint missing
  10. test_p4_output_before_and_after_reload: numerical reproducibility across save/reload
"""
import os
import copy
import hashlib
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from research.phase6_dataset import (
    GroupedBatchSampler,
    Phase6UtilityDataset,
    prepare_phase6_splits,
)
from research.phase6_model import (
    Phase6ModelConfig,
    ResidualContextModel,
    ContextAwareTwoHeadMLP,
    Phase6Loss,
    FrozenContextPredictor,
    create_ablation_variant,
)
from research.phase6_artifact import (
    validate_phase6_checkpoint,
    create_phase6_manifest,
    REQUIRED_CHECKPOINT_FIELDS,
)
from research.phase6_context import PHASE6_FEATURE_DIM
from research.utility_models import TwoHeadMLP


# ==============================================================================
# VIỆC 1: Canonical Group Identity
# ==============================================================================
class TestGroupIdentity:
    def test_group_identity_is_exact(self):
        """VIỆC 1: Verify group identity g = (scene, frame, sorted(context_ids))."""
        def make_group_key(scene, frame, context_ids):
            return (str(scene), int(frame), tuple(sorted(context_ids)))

        # S1 = {1, 2, 3} vs S2 = {3, 2, 1} -> SAME group
        k1 = make_group_key("tum_fr1_desk", 10, [1, 2, 3])
        k2 = make_group_key("tum_fr1_desk", 10, [3, 2, 1])
        assert k1 == k2, "Context ordering must be canonicalized by sorted()"

        # S1 = {1, 2, 3} vs S3 = {1, 2, 4} -> DIFFERENT group
        k3 = make_group_key("tum_fr1_desk", 10, [1, 2, 4])
        assert k1 != k3, "Different context elements must produce different groups"

        # Frame sensitivity: frame 10 vs frame 20 -> DIFFERENT group
        k4 = make_group_key("tum_fr1_desk", 20, [1, 2, 3])
        assert k1 != k4, "Different frames must produce different groups"

        # Scene sensitivity: tum_fr1 vs tum_fr2 -> DIFFERENT group
        k5 = make_group_key("tum_fr2_xyz", 10, [1, 2, 3])
        assert k1 != k5, "Different scenes must produce different groups"

        # Empty context identity
        k_empty1 = make_group_key("tum_fr1_desk", 10, [])
        k_empty2 = make_group_key("tum_fr1_desk", 10, [])
        assert k_empty1 == k_empty2
        assert len(k_empty1[2]) == 0


# ==============================================================================
# VIỆC 2: GroupedBatchSampler Invariant
# ==============================================================================
class TestGroupedBatchSampler:
    def test_grouped_batch_sampler_no_split(self):
        """VIỆC 2: Guarantees candidate groups are 100% intact and never split across batches."""
        # Create group assignments: group 0 (3 items), group 1 (5 items),
        # group 2 (12 items - oversized), group 3 (4 items), group 4 (6 items)
        group_specs = [(0, 3), (1, 5), (2, 12), (3, 4), (4, 6)]
        group_ids = []
        for gid, count in group_specs:
            group_ids.extend([gid] * count)
        group_ids = np.array(group_ids)
        total_items = len(group_ids)

        sampler = GroupedBatchSampler(group_ids, max_batch_size=10, shuffle=True, seed=42)

        seen_indices = set()
        for batch in sampler:
            batch_indices = set(batch)
            assert len(batch_indices) == len(batch), "No duplicates within a batch"
            seen_indices.update(batch_indices)

            # Check for every group in this batch: all of its members must be in this batch!
            batch_gids = set(group_ids[batch])
            for gid in batch_gids:
                all_g_members = set(np.where(group_ids == gid)[0])
                assert all_g_members.issubset(batch_indices), (
                    f"Group {gid} was split! Expected all {len(all_g_members)} members in batch, "
                    f"got {len(all_g_members.intersection(batch_indices))}"
                )

        assert seen_indices == set(range(total_items)), "All dataset items must be sampled exactly once per epoch"

    def test_grouped_batch_sampler_single_group_mode(self):
        """Verify max_batch_size=None or 1 yields exactly 1 group per batch."""
        group_ids = np.array([0, 0, 1, 1, 1, 2, 2])
        sampler = GroupedBatchSampler(group_ids, max_batch_size=None, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 3
        assert batches[0] == [0, 1]
        assert batches[1] == [2, 3, 4]
        assert batches[2] == [5, 6]


# ==============================================================================
# VIỆC 3 & 4: Phase6Loss Group-Aware Ranking & Tie-Filtering & Listwise KL
# ==============================================================================
class TestPhase6LossHardening:
    def test_phase6_loss_requires_group_ids_and_tie_filtering(self):
        """VIỆC 3: Phase6Loss raises ValueError if group_ids is None and filters ties |ΔU| <= tau."""
        loss_fn = Phase6Loss(require_group_ids=True, tau_factor=0.05)

        pred_q = torch.zeros(6)
        pred_t = torch.ones(6)
        pred_u = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        target_q = torch.zeros(6)
        target_t = torch.ones(6)
        target_u = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

        # Must raise if group_ids is None
        with pytest.raises(ValueError, match="requires group_ids"):
            loss_fn(pred_q, pred_t, pred_u, target_q, target_t, target_u, group_ids=None)

        # Test tie filtering: all pairs within each group have equal targets (diff = 0 <= tau)
        tied_target_u = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        out = loss_fn(
            pred_q, pred_t, pred_u, target_q, target_t, tied_target_u,
            group_ids=group_ids
        )
        assert out["fraction_pairs_retained"] == 0.0, "All ties should be filtered out by tau"
        assert out["loss_r"].item() == 0.0

        # Test meaningful differences: |ΔU| > tau
        distinct_target_u = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        out_distinct = loss_fn(
            pred_q, pred_t, pred_u, target_q, target_t, distinct_target_u,
            group_ids=group_ids
        )
        assert out_distinct["fraction_pairs_retained"] > 0.0
        assert out_distinct["n_groups_used_pairwise"] == 2

    def test_listwise_kl_loss_numerical(self):
        """VIỆC 4: KL ≈ 0 for matching rankings, KL > 0 for inverted rankings."""
        loss_fn = Phase6Loss(
            lambda_q=0.0, lambda_c=0.0, lambda_r=0.0,
            lambda_list=1.0, lambda_res=0.0, lambda_zero=0.0,
            tau_list=0.1
        )

        n = 5
        group_ids = torch.zeros(n, dtype=torch.long)
        target_u = torch.linspace(0.1, 1.0, n)
        dummy_q = torch.zeros(n)
        dummy_t = torch.ones(n)

        # 1. Perfectly matched predictions: pred_u == target_u
        pred_u_matched = target_u.clone()
        out_matched = loss_fn(
            dummy_q, dummy_t, pred_u_matched,
            dummy_q, dummy_t, target_u,
            group_ids=group_ids
        )
        assert out_matched["loss_list"].item() < 1e-4, (
            f"KL divergence for identical distributions should be ~0, got {out_matched['loss_list'].item()}"
        )

        # 2. Inverted predictions: pred_u is reversed target_u
        pred_u_inverted = torch.flip(target_u, dims=[0])
        out_inverted = loss_fn(
            dummy_q, dummy_t, pred_u_inverted,
            dummy_q, dummy_t, target_u,
            group_ids=group_ids
        )
        assert out_inverted["loss_list"].item() > 0.1, (
            f"KL divergence for inverted distribution should be significantly > 0, got {out_inverted['loss_list'].item()}"
        )


# ==============================================================================
# VIỆC 5 & 6: Residual Target Invariant & Empty Context Identity
# ==============================================================================
class TestResidualInvariants:
    def test_residual_target_invariant(self):
        """VIỆC 5: r* = U*(i|S) - U*(i|∅) and r*(∅) = 0."""
        # Create mock samples
        u_empty = 0.05
        u_cond = 0.08
        mock_samples = [
            {
                "scene": "tum_fr1_desk",
                "frame": 10,
                "candidate_id": 1,
                "context_ids": [],
                "context_size": 0,
                "context_type": "empty",
                "full_feature_vector": [0.1] * 32,
                "delta_q_conditional": 0.005,
                "delta_t_conditional_ms": 10.0,
                "utility_conditional": u_empty,
                "delta_q_single": 0.005,
                "delta_t_single": 10.0,
                "utility_single": u_empty,
                "split": "train",
            },
            {
                "scene": "tum_fr1_desk",
                "frame": 10,
                "candidate_id": 1,
                "context_ids": [2, 3],
                "context_size": 2,
                "context_type": "spatial_knn",
                "full_feature_vector": [0.1] * 32,
                "delta_q_conditional": 0.008,
                "delta_t_conditional_ms": 10.0,
                "utility_conditional": u_cond,
                "delta_q_single": 0.005,
                "delta_t_single": 10.0,
                "utility_single": u_empty,
                "split": "train",
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            import json
            json.dump(mock_samples, f)
            tmp_path = f.name

        try:
            train_ds, val_ds, test_ds, _ = prepare_phase6_splits(
                dataset_paths=[tmp_path],
                variant="all_features",
            )
            assert len(train_ds) == 2

            # Sample 0 is empty context: target_r must be exactly 0.0, is_empty must be 1.0
            s0 = train_ds[0]
            assert s0["is_empty"].item() == 1.0
            assert abs(s0["target_r"].item()) < 1e-8, "Empty context must have target_r == 0.0"

            # Sample 1 is conditional context: target_r = u_cond - u_empty = 0.03
            s1 = train_ds[1]
            assert s1["is_empty"].item() == 0.0
            assert abs(s1["target_r"].item() - (u_cond - u_empty)) < 1e-6
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_empty_context_identity_p4_vs_p6(self):
        """VIỆC 6: |U_hat_P6(i|∅) - U_hat_P4(i)| < 1e-6."""
        p4_model = TwoHeadMLP(in_features=11, hidden_dim=64)
        config = create_ablation_variant("all_features")
        model = ResidualContextModel(config, p4_model=p4_model)
        model.eval()

        # Generate inputs where context features are zero (empty context)
        x = torch.zeros(12, PHASE6_FEATURE_DIM)
        x[:, :11] = torch.randn(12, 11)  # random self features

        with torch.no_grad():
            _, _, u_p4 = model.p4_model(x[:, :11])
            _, _, u_p6, r = model(x, return_residual=True)

        # Residual head is initialized to zero at output
        max_error = (u_p6 - u_p4).abs().max().item()
        assert max_error < 1e-6, f"U_hat_P6(i|∅) must equal U_hat_P4(i), max error: {max_error}"
        assert r.abs().max().item() < 1e-6, "Residual correction r for empty context must be ~0"


# ==============================================================================
# VIỆC 7 & 8: P4 Backbone Immutability & Checkpoint Schema phase6-v2
# ==============================================================================
class TestBackboneAndCheckpointHardening:
    def test_p4_freeze_and_sha256_invariant(self):
        """VIỆC 7: P4 backbone parameter tensors are strictly immutable during training."""
        p4_model = TwoHeadMLP(in_features=11, hidden_dim=64)
        config = create_ablation_variant("all_features")
        model = ResidualContextModel(config, p4_model=p4_model)

        def hash_tensors(m):
            h = hashlib.sha256()
            for k, p in sorted(m.p4_model.state_dict().items()):
                h.update(k.encode())
                h.update(p.cpu().numpy().tobytes())
            return h.hexdigest()

        hash_initial = hash_tensors(model)

        # Train residual parameters only
        opt = torch.optim.Adam(model.context_parameters(), lr=1e-2)
        loss_fn = Phase6Loss()

        for _ in range(10):
            opt.zero_grad()
            x = torch.randn(8, PHASE6_FEATURE_DIM)
            target_q = torch.randn(8) * 1e-4
            target_t = torch.rand(8) * 10.0 + 1.0
            target_u = target_q / target_t
            g_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

            dq, dt, u, r = model(x, return_residual=True)
            loss_dict = loss_fn(dq, dt, u, target_q, target_t, target_u, group_ids=g_ids)
            loss_dict["total"].backward()

            # Ensure all P4 gradients are strictly None
            for p in model.p4_model.parameters():
                assert p.grad is None
                assert not p.requires_grad

            opt.step()

        hash_final = hash_tensors(model)
        assert hash_initial == hash_final, "P4 parameter SHA256 hash must be 100% invariant!"

    def test_checkpoint_schema_phase6_v2(self):
        """VIỆC 8: Checkpoint validation enforces schema_version 'phase6-v2' and all fields."""
        valid_ckpt = {
            "schema_version": "phase6-v2",
            "architecture": "residual_context",
            "variant": "self_neighbor_selected",
            "seed": 42,
            "protocol_version": "v1",
            "model_state": {"layer.weight": torch.tensor([1.0])},
            "config": {"hidden_dim": 64},
            "normalizer_path": "normalizer.pt",
            "p4_checkpoint": "p4_seed_42.pt",
            "p4_frozen": True,
            "git_commit": "abcdef123",
        }
        is_valid, errors = validate_phase6_checkpoint(valid_ckpt)
        assert is_valid, f"Expected valid checkpoint, got errors: {errors}"

        # 1. Missing required field
        invalid_ckpt_missing = copy.deepcopy(valid_ckpt)
        del invalid_ckpt_missing["schema_version"]
        is_valid, errors = validate_phase6_checkpoint(invalid_ckpt_missing)
        assert not is_valid
        assert any("schema_version" in e for e in errors)

        # 2. Outdated schema version
        invalid_ckpt_ver = copy.deepcopy(valid_ckpt)
        invalid_ckpt_ver["schema_version"] = "v1"
        is_valid, errors = validate_phase6_checkpoint(invalid_ckpt_ver)
        assert not is_valid
        assert any("Expected 'phase6-v2'" in e for e in errors)

        # 3. Residual without frozen P4
        invalid_ckpt_frozen = copy.deepcopy(valid_ckpt)
        invalid_ckpt_frozen["p4_frozen"] = False
        is_valid, errors = validate_phase6_checkpoint(invalid_ckpt_frozen)
        assert not is_valid
        assert any("p4_frozen must be True" in e for e in errors)


# ==============================================================================
# VIỆC 9 & 10: FrozenContextPredictor Failure Mode & Reload Numerical Reproducibility
# ==============================================================================
class TestPredictorAndReloadReproducibility:
    def test_frozen_context_predictor_requires_p4(self):
        """VIỆC 9: FrozenContextPredictor fails loudly if P4 checkpoint missing for residual model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from research.phase6_dataset import Phase6FeatureNormalizer
            norm_path = os.path.join(tmp_dir, "norm.json")
            Phase6FeatureNormalizer().fit(np.zeros((2, 32))).save_json(norm_path)

            model_ckpt_path = os.path.join(tmp_dir, "model.pt")
            torch.save({
                "schema_version": "phase6-v2",
                "architecture": "residual_context",
                "variant": "all_features",
                "seed": 42,
                "protocol_version": "v1",
                "model_state": {},
                "config": Phase6ModelConfig().to_dict(),
                "normalizer_path": norm_path,
                "p4_checkpoint": os.path.join(tmp_dir, "non_existent_p4.pt"),
                "p4_frozen": True,
                "git_commit": "abc",
            }, model_ckpt_path)

            # Should raise RuntimeError when p4_ckpt doesn't exist
            with pytest.raises(RuntimeError, match="requires a valid pretrained Phase 4 checkpoint"):
                FrozenContextPredictor(
                    checkpoint_path=model_ckpt_path,
                    normalizer_path=norm_path,
                    device="cpu",
                )

    def test_p4_output_before_and_after_reload(self):
        """VIỆC 10: P4 predictions match exactly before and after checkpoint reload |U_train - U_reload| < 1e-6."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Create and save P4 checkpoint
            p4_path = os.path.join(tmp_dir, "p4.pt")
            p4_net = TwoHeadMLP(in_features=11, hidden_dim=64)
            torch.save({"model_state": p4_net.state_dict()}, p4_path)

            # 2. Build ResidualContextModel
            config = create_ablation_variant("all_features")
            model = ResidualContextModel(config, p4_model=p4_net)
            model.eval()

            test_x = torch.randn(10, PHASE6_FEATURE_DIM)
            with torch.no_grad():
                dq_before, dt_before, u_before = model(test_x)

            # 3. Save Phase 6 checkpoint
            p6_path = os.path.join(tmp_dir, "p6.pt")
            torch.save({
                "schema_version": "phase6-v2",
                "architecture": "residual_context",
                "variant": "all_features",
                "seed": 42,
                "protocol_version": "v1",
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "normalizer_path": "",
                "p4_checkpoint": p4_path,
                "p4_frozen": True,
                "git_commit": "abc",
            }, p6_path)

            # 4. Reload model via state_dict and identical P4
            p4_reload = TwoHeadMLP(in_features=11, hidden_dim=64)
            p4_reload.load_state_dict(torch.load(p4_path, weights_only=False)["model_state"])
            p4_reload.eval()
            for p in p4_reload.parameters():
                p.requires_grad = False

            model_reloaded = ResidualContextModel(config, p4_model=p4_reload)
            ckpt = torch.load(p6_path, weights_only=False)
            model_reloaded.load_state_dict(ckpt["model_state"])
            model_reloaded.eval()

            with torch.no_grad():
                dq_after, dt_after, u_after = model_reloaded(test_x)

            assert torch.allclose(u_before, u_after, atol=1e-6), "Utility predictions must match exactly after reload"
            assert torch.allclose(dq_before, dq_after, atol=1e-6), "Quality predictions must match exactly after reload"
            assert torch.allclose(dt_before, dt_after, atol=1e-6), "Cost predictions must match exactly after reload"


# ==============================================================================
# VIỆC 11: Exact-Context Rank Stability Invariant
# ==============================================================================
class TestExactRankStability:
    def test_exact_context_rank_stability_invariants(self):
        """VIỆC 11: Verify exact context rank stability schema and semantic integrity."""
        artifact_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "phase6_context_utility", "rank_stability_analysis.json"
        )
        if not os.path.exists(artifact_path):
            pytest.skip("rank_stability_analysis.json not yet generated")

        import json
        with open(artifact_path, "r") as f:
            data = json.load(f)

        summary = data["overall_summary"]
        per_group = data["per_group_results"]

        # Semantic invariant: n_total_evaluated_groups must equal number of exact group records
        assert summary["n_total_evaluated_groups"] == len(per_group), (
            f"n_total_evaluated_groups ({summary['n_total_evaluated_groups']}) must equal len(per_group) ({len(per_group)})"
        )
        assert len(per_group) >= 500, f"Expected >= 500 exact context groups, got {len(per_group)}"

        # Validate fields for each exact group
        for g in per_group[:50]:
            assert "context_ids" in g, "Each group record must contain explicit context_ids"
            assert isinstance(g["context_ids"], list), "context_ids must be a list"
            assert "n_candidates" in g, "Must record n_candidates in candidate pool"
            assert g["n_candidates"] >= 3, "Pool size must be >= 3 for valid ranking"
            assert "mean_candidate_context_overlap" in g, "Must record candidate-context overlap"
            assert -1.0 <= g["spearman_rho"] <= 1.0, "Spearman rho must be in [-1, 1]"
            assert 0.0 <= g["overlap_at_5"] <= 1.0, "Overlap@5 must be in [0, 1]"

        # Scientific validity: rank stability is substantial (> 0.50)
        assert summary["mean_spearman_rho"] > 0.50, "Conditional utility must exhibit substantial rank stability"

