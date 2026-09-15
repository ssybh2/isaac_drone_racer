"""Re-evaluate OpenVINS fault-isolation traces against truth at the VIO timestamp.

The runtime trace records current simulator truth together with an OpenVINS
estimate that can lag simulator time. Comparing those states directly can
inflate position/orientation errors during oscillatory motion. This script uses
``odom_age_s`` to recover each estimator timestamp and interpolates simulator
truth to that same time before recomputing error metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize(q0)
    q1 = _normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize(q0 + float(alpha) * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - float(alpha)) * theta) / sin_theta
    w1 = math.sin(float(alpha) * theta) / sin_theta
    return _normalize(w0 * q0 + w1 * q1)


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(_normalize(q0), _normalize(q1))), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(q)
    return np.array(
        [
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=np.float64,
    )


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _summary(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    return (
        f"mean={np.mean(values):.6f}  rmse={np.sqrt(np.mean(values**2)):.6f}  "
        f"p95={np.percentile(values, 95):.6f}  max={np.max(values):.6f}"
    )


def _interp_scalar(times: np.ndarray, values: np.ndarray, t: float) -> float:
    return float(np.interp(float(t), times, values))


def _interp_vector(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    return np.array([_interp_scalar(times, values[:, i], t) for i in range(values.shape[1])])


def _interp_quaternion(times: np.ndarray, quats: np.ndarray, t: float) -> np.ndarray:
    idx = int(np.searchsorted(times, float(t), side="left"))
    if idx <= 0:
        return _normalize(quats[0])
    if idx >= len(times):
        return _normalize(quats[-1])
    left = idx - 1
    right = idx
    dt = times[right] - times[left]
    if dt <= 1.0e-12:
        return _normalize(quats[right])
    alpha = float(np.clip((float(t) - times[left]) / dt, 0.0, 1.0))
    return _slerp(quats[left], quats[right], alpha)


def analyze(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Trace is empty: {path}")

    times = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    truth_p = np.array(
        [[float(r["truth_px"]), float(r["truth_py"]), float(r["truth_pz"])] for r in rows],
        dtype=np.float64,
    )
    truth_v = np.array(
        [[float(r["truth_vx"]), float(r["truth_vy"]), float(r["truth_vz"])] for r in rows],
        dtype=np.float64,
    )
    truth_q = np.array(
        [
            [float(r["truth_qw"]), float(r["truth_qx"]), float(r["truth_qy"]), float(r["truth_qz"])]
            for r in rows
        ],
        dtype=np.float64,
    )

    current_pos_err = []
    aligned_pos_err = []
    current_vel_err = []
    aligned_vel_err = []
    current_ori_err = []
    aligned_ori_err = []
    aligned_rpy_err = []
    ages = []
    estimate_times = []

    for r in rows:
        if not r.get("vio_qw") or not r.get("odom_age_s"):
            continue
        now = float(r["t_s"])
        age = float(r["odom_age_s"])
        estimate_time = now - age
        if estimate_time < times[0] - 1.0e-9 or estimate_time > times[-1] + 1.0e-9:
            continue

        vio_p = np.array([float(r["vio_px"]), float(r["vio_py"]), float(r["vio_pz"])])
        vio_v = np.array([float(r["vio_vx"]), float(r["vio_vy"]), float(r["vio_vz"])])
        vio_q = np.array([float(r["vio_qw"]), float(r["vio_qx"]), float(r["vio_qy"]), float(r["vio_qz"])])

        now_truth_p = np.array([float(r["truth_px"]), float(r["truth_py"]), float(r["truth_pz"])])
        now_truth_v = np.array([float(r["truth_vx"]), float(r["truth_vy"]), float(r["truth_vz"])])
        now_truth_q = np.array(
            [float(r["truth_qw"]), float(r["truth_qx"]), float(r["truth_qy"]), float(r["truth_qz"])]
        )

        aligned_truth_p = _interp_vector(times, truth_p, estimate_time)
        aligned_truth_v = _interp_vector(times, truth_v, estimate_time)
        aligned_truth_q = _interp_quaternion(times, truth_q, estimate_time)

        current_pos_err.append(float(np.linalg.norm(vio_p - now_truth_p)))
        aligned_pos_err.append(float(np.linalg.norm(vio_p - aligned_truth_p)))
        current_vel_err.append(float(np.linalg.norm(vio_v - now_truth_v)))
        aligned_vel_err.append(float(np.linalg.norm(vio_v - aligned_truth_v)))
        current_ori_err.append(_quat_angle_deg(vio_q, now_truth_q))
        aligned_ori_err.append(_quat_angle_deg(vio_q, aligned_truth_q))
        aligned_rpy_err.append(np.degrees(_wrap(_quat_to_rpy(vio_q) - _quat_to_rpy(aligned_truth_q))))
        ages.append(age)
        estimate_times.append(estimate_time)

    if not ages:
        raise ValueError(f"No valid OpenVINS rows with odom_age_s in {path}")

    current_pos_err = np.asarray(current_pos_err)
    aligned_pos_err = np.asarray(aligned_pos_err)
    current_vel_err = np.asarray(current_vel_err)
    aligned_vel_err = np.asarray(aligned_vel_err)
    current_ori_err = np.asarray(current_ori_err)
    aligned_ori_err = np.asarray(aligned_ori_err)
    aligned_rpy_err = np.asarray(aligned_rpy_err)
    ages = np.asarray(ages)
    estimate_times = np.asarray(estimate_times)

    print("\n" + "=" * 84)
    print(path)
    print("=" * 84)
    print("odom age [s]          :", _summary(ages))
    print("current position [m]  :", _summary(current_pos_err))
    print("aligned position [m]  :", _summary(aligned_pos_err))
    print("current velocity [m/s]:", _summary(current_vel_err))
    print("aligned velocity [m/s]:", _summary(aligned_vel_err))
    print("current orientation [deg]:", _summary(current_ori_err))
    print("aligned orientation [deg]:", _summary(aligned_ori_err))

    print("\nTimestamp-aligned RPY absolute error [deg]")
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        values = np.abs(aligned_rpy_err[:, i])
        print(f"  {axis:5s}: {_summary(values)}")

    start = estimate_times[0]
    print("\nTimestamp-aligned checkpoints after first valid VIO sample")
    for checkpoint in (1.0, 5.0, 10.0, 20.0, 30.0):
        idx = int(np.argmin(np.abs((estimate_times - start) - checkpoint)))
        if abs((estimate_times[idx] - start) - checkpoint) <= 0.15:
            print(
                f"  {checkpoint:>4.0f}s: pos={aligned_pos_err[idx]:.6f} m  "
                f"vel={aligned_vel_err[idx]:.6f} m/s  ori={aligned_ori_err[idx]:.6f} deg"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    args = parser.parse_args()
    for trace in args.traces:
        analyze(trace.expanduser().resolve())


if __name__ == "__main__":
    main()
