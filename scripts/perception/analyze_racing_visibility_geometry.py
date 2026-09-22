"""Offline geometry audit for a collected high-speed racing vision dataset.

This script does not run Isaac Sim.  It reads the exact camera/world transforms
stored in Stage2 label JSON files and answers two questions:

1. Why does the ACTIVE mission gate have fewer than two visible corners?
2. Would ANY gate in the known Circular-12 global map have supplied a usable
   >=2-corner visual update on the same frame?

The audit uses the production pinhole camera matrix stored in each label and the
authoritative gate opening geometry from assets/gate/gate_keypoints.json.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from perception.stage2_calibration import load_stage2_gate_geometry


def _matrix(payload: dict, name: str) -> np.ndarray:
    item = payload["truth"][name]
    if item is None:
        raise ValueError(f"{name} is missing from dataset label")
    T = np.asarray(item["matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    return T


def _circular12_gate_poses() -> list[np.ndarray]:
    """Return actor-frame T_wg for the committed radius-12 Circular-12 map."""
    poses: list[np.ndarray] = []
    radius = 12.0
    cx, cy, z = 0.0, 12.0, 1.0
    for i in range(12):
        theta = -0.5 * math.pi + i * (math.pi / 6.0)
        yaw = i * (math.pi / 6.0)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T[:3, 3] = np.array([x, y, z], dtype=np.float64)
        poses.append(T)
    return poses


def _project_gate(
    *,
    T_wc: np.ndarray,
    T_wg: np.ndarray,
    points_g: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_cw = np.linalg.inv(T_wc)
    points_h = np.concatenate(
        [points_g, np.ones((points_g.shape[0], 1), dtype=np.float64)], axis=1
    )
    points_w = (T_wg @ points_h.T).T
    points_c = (T_cw @ points_w.T).T[:, :3]

    z = points_c[:, 2]
    uv = np.full((points_c.shape[0], 2), np.nan, dtype=np.float64)
    front = z > 1.0e-9
    uv[front, 0] = K[0, 0] * points_c[front, 0] / z[front] + K[0, 2]
    uv[front, 1] = K[1, 1] * points_c[front, 1] / z[front] + K[1, 2]
    visible = (
        front
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    return points_c, uv, visible


def _center_bearing_deg(
    *, T_wc: np.ndarray, T_wg: np.ndarray, center_g: np.ndarray
) -> tuple[float, float, float]:
    p_g = np.concatenate([center_g, [1.0]])
    p_c = (np.linalg.inv(T_wc) @ T_wg @ p_g)[:3]
    x, y, z = (float(v) for v in p_c)
    horizontal = math.degrees(math.atan2(x, z))
    vertical = math.degrees(math.atan2(y, math.hypot(x, z)))
    distance = float(np.linalg.norm(p_c))
    return horizontal, vertical, distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <dataset>/geometry_audit.json",
    )
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    labels: list[Path] = []
    for split in ("train", "val", "test"):
        labels.extend(sorted((root / split / "labels").glob("*.json")))
    if not labels:
        raise FileNotFoundError(f"No Stage2 labels found under {root}")

    geometry = load_stage2_gate_geometry()
    points_g = np.asarray(geometry.object_points_g, dtype=np.float64)
    center_g = np.asarray(geometry.center_g, dtype=np.float64)
    mapped_poses = _circular12_gate_poses()

    active_hist = Counter()
    any_hist = Counter()
    mapped_ge2_per_frame: list[int] = []
    h_bearings: list[float] = []
    v_bearings: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    body_rates: list[float] = []
    failure_reason = Counter()

    center_front = 0
    center_h_in = 0
    center_v_in = 0
    center_both_in = 0

    for path in labels:
        payload = json.loads(path.read_text(encoding="utf-8"))
        T_wc = _matrix(payload, "T_wc")
        T_wg_active = _matrix(payload, "T_wg")
        K = np.asarray(payload["camera"]["K"], dtype=np.float64)
        width = int(payload["camera"]["image_width"])
        height = int(payload["camera"]["image_height"])

        stored_visible = np.asarray(payload["visible"], dtype=bool)
        active_count = int(stored_visible.sum())
        active_hist[active_count] += 1

        h_deg, v_deg, distance = _center_bearing_deg(
            T_wc=T_wc,
            T_wg=T_wg_active,
            center_g=center_g,
        )
        h_bearings.append(h_deg)
        v_bearings.append(v_deg)
        distances.append(distance)

        half_h = math.degrees(math.atan2(0.5 * width, K[0, 0]))
        half_v = math.degrees(math.atan2(0.5 * height, K[1, 1]))

        p_center_c = (
            np.linalg.inv(T_wc)
            @ T_wg_active
            @ np.concatenate([center_g, [1.0]])
        )[:3]
        is_front = bool(p_center_c[2] > 0.0)
        h_in = is_front and abs(h_deg) <= half_h
        v_in = is_front and abs(v_deg) <= half_v
        center_front += int(is_front)
        center_h_in += int(h_in)
        center_v_in += int(v_in)
        center_both_in += int(h_in and v_in)

        if active_count < 2:
            if not is_front:
                failure_reason["center_behind_camera"] += 1
            elif not h_in and not v_in:
                failure_reason["center_out_horizontal_and_vertical"] += 1
            elif not h_in:
                failure_reason["center_out_horizontal_only"] += 1
            elif not v_in:
                failure_reason["center_out_vertical_only"] += 1
            else:
                failure_reason["center_inside_fov_but_lt2_corners"] += 1

        max_visible = 0
        ge2_gate_count = 0
        for T_wg in mapped_poses:
            _, _, visible = _project_gate(
                T_wc=T_wc,
                T_wg=T_wg,
                points_g=points_g,
                K=K,
                width=width,
                height=height,
            )
            count = int(visible.sum())
            max_visible = max(max_visible, count)
            ge2_gate_count += int(count >= 2)
        any_hist[max_visible] += 1
        mapped_ge2_per_frame.append(ge2_gate_count)

        extra = payload.get("extra") or {}
        speeds.append(float(extra.get("speed_mps", 0.0)))
        body_rates.append(float(extra.get("body_rate_norm_radps", 0.0)))

    n = len(labels)
    active_ge2 = sum(v for k, v in active_hist.items() if k >= 2)
    any_ge2 = sum(v for k, v in any_hist.items() if k >= 2)

    def stats(values: list[float]) -> dict:
        a = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
        }

    result = {
        "dataset": str(root),
        "frames": n,
        "active_gate": {
            "visible_corner_histogram": {
                str(k): int(active_hist.get(k, 0)) for k in range(5)
            },
            "ge2_frames": int(active_ge2),
            "ge2_fraction": float(active_ge2 / n),
        },
        "any_mapped_gate": {
            "max_visible_corner_histogram": {
                str(k): int(any_hist.get(k, 0)) for k in range(5)
            },
            "ge2_frames": int(any_ge2),
            "ge2_fraction": float(any_ge2 / n),
            "mean_number_of_mapped_gates_with_ge2": float(
                np.mean(mapped_ge2_per_frame)
            ),
            "max_number_of_mapped_gates_with_ge2": int(
                max(mapped_ge2_per_frame, default=0)
            ),
        },
        "active_gate_center": {
            "front_fraction": float(center_front / n),
            "horizontal_in_fov_fraction": float(center_h_in / n),
            "vertical_in_fov_fraction": float(center_v_in / n),
            "both_axes_in_fov_fraction": float(center_both_in / n),
            "horizontal_bearing_abs_deg": stats([abs(x) for x in h_bearings]),
            "vertical_bearing_abs_deg": stats([abs(x) for x in v_bearings]),
            "distance_m": stats(distances),
        },
        "active_lt2_failure_reason": dict(failure_reason),
        "motion": {
            "speed_mps": stats(speeds),
            "body_rate_norm_radps": stats(body_rates),
        },
    }

    output = args.output
    if output is None:
        output = root / "geometry_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=" * 96)
    print("CIRCULAR-12 RACING CAMERA / MAP GEOMETRY AUDIT")
    print("=" * 96)
    print(f"frames                         : {n}")
    print(f"active gate >=2               : {active_ge2}/{n} = {active_ge2/n:.4f}")
    print(f"ANY mapped gate >=2           : {any_ge2}/{n} = {any_ge2/n:.4f}")
    print(
        "active center horizontal FOV : "
        f"{center_h_in}/{n} = {center_h_in/n:.4f}"
    )
    print(
        "active center vertical FOV   : "
        f"{center_v_in}/{n} = {center_v_in/n:.4f}"
    )
    print(
        "active center both axes      : "
        f"{center_both_in}/{n} = {center_both_in/n:.4f}"
    )
    print("active <2 failure reasons     :", dict(failure_reason))
    print("abs horizontal bearing deg   :", result["active_gate_center"]["horizontal_bearing_abs_deg"])
    print("abs vertical bearing deg     :", result["active_gate_center"]["vertical_bearing_abs_deg"])
    print("gate distance m              :", result["active_gate_center"]["distance_m"])
    print("speed m/s                    :", result["motion"]["speed_mps"])
    print("body-rate norm rad/s         :", result["motion"]["body_rate_norm_radps"])
    print(f"wrote                         : {output}")


if __name__ == "__main__":
    main()
