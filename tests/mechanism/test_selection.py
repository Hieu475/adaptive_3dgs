import torch

from research.mechanism.selection import combined_residual_percentile


def test_combined_residual_percentile_ranks_each_modality_before_combining():
    color = torch.tensor([0.0, 0.3, 0.1, 0.2])
    depth = torch.tensor([0.4, 0.1, 0.3, 0.2])
    score = combined_residual_percentile(color, depth)
    assert torch.allclose(score, torch.tensor([1.0, 1.0, 1.0, 1.0]))
    # Each item is high in one residual modality and low in the other.


def test_combined_residual_percentile_prefers_high_rank_in_both_channels():
    color = torch.tensor([0.0, 0.1, 0.9, 1.0])
    depth = torch.tensor([0.0, 0.1, 0.9, 1.0])
    score = combined_residual_percentile(color, depth)
    assert torch.all(score[1:] > score[:-1])


def test_nonfinite_residuals_receive_lowest_stable_rank():
    color = torch.tensor([0.2, float("nan"), 0.8])
    depth = torch.tensor([0.2, 0.9, 0.8])
    score = combined_residual_percentile(color, depth)
    assert score[1] < score[2]
    assert torch.isfinite(score).all()


def test_combined_residual_percentile_rejects_mismatched_shapes():
    try:
        combined_residual_percentile(torch.ones(2), torch.ones(3))
    except ValueError as error:
        assert "matching 1D" in str(error)
    else:
        raise AssertionError("expected mismatched residual tensors to fail")
