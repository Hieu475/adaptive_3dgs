"""Tests for TUM dataset data association and per-pair timestamp synchronization."""
from pathlib import Path
import pytest
from datasets.tum_dataset import TUMDataset


def test_tum_strict_per_pair_association():
    """Verify that every associated RGB-D pair has |t_rgb - t_d| <= 50ms."""
    tum_path = Path('datasets/TUM/rgbd_dataset_freiburg1_desk')
    if not tum_path.exists():
        pytest.skip("TUM fr1/desk dataset not present on disk")

    ds = TUMDataset(tum_path, max_frames=50)
    stats = ds.get_association_statistics()

    assert stats['fraction_above_50ms'] == 0.0, f"Found pairs exceeding 50ms: {stats['fraction_above_50ms']*100}%"
    assert stats['max_ms'] <= 50.0, f"Max diff exceeds 50ms: {stats['max_ms']} ms"
    assert stats['mean_ms'] < 25.0, f"Mean diff unexpectedly high: {stats['mean_ms']} ms"
