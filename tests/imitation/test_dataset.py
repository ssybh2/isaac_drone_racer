import numpy as np
import pytest

from imitation.dataset import concatenate_datasets


def _dataset(offset: float, episode: int) -> dict[str, np.ndarray]:
    return {
        "observation": np.full((3, 31), offset, dtype=np.float32),
        "expert_action": np.full((3, 4), offset, dtype=np.float32),
        "episode_id": np.full(3, episode, dtype=np.int32),
        "episode_step": np.arange(3, dtype=np.int32),
        "target_speed_mps": np.asarray(14.0, dtype=np.float32),
        "schema_version": np.asarray("demo-v1"),
    }


def test_concatenate_datasets_ignores_scalar_metadata_and_offsets_episodes():
    merged = concatenate_datasets(_dataset(1.0, 0), _dataset(2.0, 0))
    assert merged["observation"].shape == (6, 31)
    assert merged["expert_action"].shape == (6, 4)
    assert "target_speed_mps" not in merged
    assert "schema_version" not in merged
    np.testing.assert_array_equal(
        merged["episode_id"],
        np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
    )
