"""Dataset helpers shared by expert collection, BC and DAgger."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = (
    "observation",
    "expert_action",
    "episode_id",
    "episode_step",
)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for key in REQUIRED_ARRAYS:
        if key not in result:
            raise ValueError(f"dataset missing required array {key!r}")
    if (
        result["observation"].ndim != 2
        or result["observation"].shape[1] != 31
    ):
        raise ValueError("observation must have shape [N, 31]")
    if (
        result["expert_action"].ndim != 2
        or result["expert_action"].shape[1] != 4
    ):
        raise ValueError("expert_action must have shape [N, 4]")
    if len(result["observation"]) != len(result["expert_action"]):
        raise ValueError("observation/expert_action sample counts differ")
    return result


def save_dataset(
    path: str | Path, arrays: dict[str, np.ndarray]
) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def concatenate_datasets(
    *datasets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Concatenate only sample-aligned arrays and offset episode IDs.

    Scalar metadata such as target_speed_mps and schema_version is deliberately
    not concatenated. This keeps dataset metadata separate from per-step arrays.
    """
    if not datasets:
        raise ValueError("at least one dataset is required")

    sample_counts = [int(len(dataset["observation"])) for dataset in datasets]
    common = set(datasets[0].keys())
    for current in datasets[1:]:
        common &= set(current.keys())

    sample_keys = []
    for key in sorted(common):
        aligned = True
        for current, count in zip(datasets, sample_counts):
            array = np.asarray(current[key])
            if array.ndim < 1 or int(array.shape[0]) != count:
                aligned = False
                break
        if aligned:
            sample_keys.append(key)

    for required in REQUIRED_ARRAYS:
        if required not in sample_keys:
            raise ValueError(
                f"required sample-aligned array {required!r} is missing"
            )

    normalized: list[dict[str, np.ndarray]] = []
    episode_offset = 0
    for current in datasets:
        item = {
            key: np.asarray(current[key])
            for key in sample_keys
        }
        episode_id = item["episode_id"].astype(np.int64, copy=True)
        if episode_id.size:
            episode_id -= int(episode_id.min())
            episode_id += episode_offset
            episode_offset = int(episode_id.max()) + 1
        item["episode_id"] = episode_id.astype(np.int32)
        normalized.append(item)

    return {
        key: np.concatenate(
            [current[key] for current in normalized],
            axis=0,
        )
        for key in sample_keys
    }
