"""Rebuild manifests for an already captured diversified color dataset.

This script does not launch Isaac Sim and never rewrites images or labels.
It is intended for recovery when capture completed but Isaac Sim shutdown
occurred before the collector flushed manifest.json files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def _sorted_unique(values):
    return sorted({float(value) for value in values})


def _scan_split(root: Path, split: str) -> tuple[dict, list[dict]]:
    label_dir = root / "vision" / split / "labels"
    image_dir = root / "vision" / split / "images"
    label_paths = sorted(label_dir.glob("*.json"))
    image_paths = sorted(image_dir.glob("*.png"))
    if len(label_paths) != len(image_paths):
        raise RuntimeError(
            f"{split}: image/label count mismatch "
            f"{len(image_paths)} != {len(label_paths)}"
        )
    if not label_paths:
        raise RuntimeError(f"{split}: no labels found under {label_dir}")

    visible_hist = Counter()
    by_run: dict[int, list[dict]] = defaultdict(list)
    camera_pitch_values = []
    body_height_values = []

    for path in label_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_split") != split:
            raise RuntimeError(
                f"{path}: dataset_split={payload.get('dataset_split')!r}, "
                f"expected {split!r}"
            )
        mapped = payload.get("mapped_gates") or []
        best_visible = max(
            (
                int(sum(bool(v) for v in gate.get("visible", [])))
                for gate in mapped
            ),
            default=0,
        )
        visible_hist[best_visible] += 1
        run_index = int(payload["run_index"])
        by_run[run_index].append(payload)
        camera_pitch_values.append(float(payload["camera_pitch_up_deg"]))
        body_height_values.append(float(payload["body_height_m"]))

    runs = []
    for run_index in sorted(by_run):
        frames = sorted(
            by_run[run_index],
            key=lambda item: int(item["frame_index"]),
        )
        first = frames[0]
        last = frames[-1]
        runs.append(
            {
                "run_index": int(run_index),
                "split": split,
                "speed_mps": float(first["speed_mps"]),
                "path_radius_m": float(first["path_radius_m"]),
                "phase_offset_deg": float(first["phase_offset_deg"]),
                "frames": len(frames),
                "duration_s": float(last["t_s"]),
                "bank_deg": float(first["prescribed_bank_deg"]),
                "camera_pitch_up_deg": float(first["camera_pitch_up_deg"]),
                "body_height_m": float(first["body_height_m"]),
            }
        )

    pitch_unique = _sorted_unique(camera_pitch_values)
    height_unique = _sorted_unique(body_height_values)
    if len(pitch_unique) != 1 or len(height_unique) != 1:
        raise RuntimeError(
            f"{split}: non-uniform camera/height values "
            f"pitch={pitch_unique}, height={height_unique}"
        )

    manifest = {
        "schema": "isaac_drone_racer.circular12_diversified_color_split.v1",
        "split": split,
        "samples": len(label_paths),
        "runs": runs,
        "run_count": len(runs),
        "camera_pitch_up_deg": pitch_unique[0],
        "body_height_m": height_unique[0],
        "any_mapped_gate_max_visible_corner_histogram": {
            str(k): int(visible_hist.get(k, 0)) for k in range(5)
        },
        "split_contract": (
            "whole (speed_mps, path_radius_m) conditions; "
            "all phase offsets stay in one split"
        ),
        "recovered_from_existing_labels": True,
    }
    return manifest, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "artifacts/racing_vision/"
            "circular12_color20_h207_diversified_v1"
        ),
    )
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    split_manifests = {}
    all_runs = []
    for split in SPLITS:
        manifest, runs = _scan_split(root, split)
        split_manifests[split] = manifest
        all_runs.extend(runs)
        path = root / "vision" / split / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"[finalize] {split}: samples={manifest['samples']} "
            f"runs={manifest['run_count']} -> {path}",
            flush=True,
        )

    camera_values = _sorted_unique(
        manifest["camera_pitch_up_deg"]
        for manifest in split_manifests.values()
    )
    height_values = _sorted_unique(
        manifest["body_height_m"]
        for manifest in split_manifests.values()
    )
    if len(camera_values) != 1 or len(height_values) != 1:
        raise RuntimeError(
            f"dataset has inconsistent camera/height values: "
            f"pitch={camera_values}, height={height_values}"
        )

    condition_owner = {}
    for run in all_runs:
        key = (float(run["speed_mps"]), float(run["path_radius_m"]))
        owner = condition_owner.setdefault(key, run["split"])
        if owner != run["split"]:
            raise RuntimeError(
                f"split leakage: condition {key} appears in "
                f"{owner} and {run['split']}"
            )

    root_manifest = {
        "schema": "isaac_drone_racer.circular12_diversified_color_dataset.v1",
        "purpose": "12-class Gate-ID + 4-corner visual retraining",
        "control_source": "prescribed_coordinated_circle_not_policy",
        "track": "CIRCULAR_12_GATE_TRACK_CONFIG",
        "track_gate_radius_m": 12.0,
        "track_center_xy_m": [0.0, 12.0],
        "camera_pitch_up_deg": camera_values[0],
        "body_height_m": height_values[0],
        "speeds_mps": _sorted_unique(run["speed_mps"] for run in all_runs),
        "path_radii_m": _sorted_unique(
            run["path_radius_m"] for run in all_runs
        ),
        "phase_offsets_deg": _sorted_unique(
            run["phase_offset_deg"] for run in all_runs
        ),
        "total_runs": len(all_runs),
        "split_run_counts": {
            split: int(split_manifests[split]["run_count"])
            for split in SPLITS
        },
        "split_sample_counts": {
            split: int(split_manifests[split]["samples"])
            for split in SPLITS
        },
        "total_samples": sum(
            int(split_manifests[split]["samples"]) for split in SPLITS
        ),
        "leakage_guard": (
            "speed/radius condition groups are disjoint across train/val/test"
        ),
        "gate_identity": (
            "high-contrast gate texture -> supervised Gate ID 1..12; "
            "known-map reprojection remains runtime geometric consistency gate"
        ),
        "recovered_from_existing_labels": True,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(root_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 96)
    print("DIVERSIFIED DATASET MANIFESTS FINALIZED")
    print("=" * 96)
    print(json.dumps(root_manifest, indent=2))
    print(f"[finalize] root manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
