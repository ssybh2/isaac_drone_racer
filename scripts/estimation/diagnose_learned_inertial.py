"""Targeted learned-inertial estimator diagnostics.

Runs five modes on the same replay trajectory, each in a fresh Isaac process:

A: IMU only
S: IMU + TCN shadow (network runs at 20 Hz but is not fused)
B: IMU + fused 20 Hz TCN
P: IMU + fused 20 Hz TCN + Gate-PnP position only
C: IMU + fused 20 Hz TCN + Gate-PnP position + orientation

This separates online TCN prediction quality from EKF fusion and isolates the
effect of the gate orientation measurement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "circle", "lissajous"),
    default="translate_x",
)
parser.add_argument("--amplitude_m", type=float, default=1.5)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--progress-every", type=int, default=100)
parser.add_argument("--imu-noise-seed", type=int, default=0)
parser.add_argument("--imu-accel-white-noise-sigma-mps2", type=float, default=0.0)
parser.add_argument("--imu-gyro-white-noise-sigma-radps", type=float, default=0.0)
parser.add_argument("--imu-accel-initial-bias-sigma-mps2", type=float, default=0.0)
parser.add_argument("--imu-gyro-initial-bias-sigma-radps", type=float, default=0.0)
parser.add_argument("--imu-accel-bias-rw-sigma-mps2-sqrt-s", type=float, default=0.0)
parser.add_argument("--imu-gyro-bias-rw-sigma-radps-sqrt-s", type=float, default=0.0)
parser.add_argument("--ekf-accel-noise-sigma", type=float, default=None)
parser.add_argument("--ekf-gyro-noise-sigma", type=float, default=None)
parser.add_argument("--ekf-accel-bias-rw-sigma", type=float, default=None)
parser.add_argument("--ekf-gyro-bias-rw-sigma", type=float, default=None)
parser.add_argument(
    "--learned-fusion-rate-hz",
    type=float,
    default=None,
    help="Fuse only a subset of learned predictions while keeping prediction rate unchanged.",
)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-v0",
)
parser.add_argument(
    "--learned-checkpoint",
    type=Path,
    default=Path("artifacts/imo_tcn/model_v1.pt"),
)
parser.add_argument(
    "--gate-checkpoint",
    type=Path,
    default=Path(
        "artifacts/stage2_next_steps_20260911/checkpoints/"
        "torchvision_keypointrcnn_best.pt"
    ),
)
parser.add_argument(
    "--visibility-checkpoint",
    type=Path,
    default=Path(
        "artifacts/stage2_next_steps_20260911/checkpoints/"
        "gate_keypoint_net_best.pt"
    ),
)
parser.add_argument("--disable-visibility", action="store_true")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/learned_inertial_diagnostics"),
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


MODE_LABELS = {
    "A": "imu_only",
    "S": "imu_tcn_shadow",
    "B": "imu_tcn_20hz",
    "P": "imu_tcn_gate_position_only",
    "C": "imu_tcn_gate_pose",
}


def _worker_command(mode: str, output_dir: Path, replay_npz: Path) -> list[str]:
    worker = Path(__file__).with_name("evaluate_learned_inertial_mode.py")
    command = [
        sys.executable,
        str(worker),
        "--mode", mode,
        "--steps", str(args.steps),
        "--profile", args.profile,
        "--amplitude_m", str(args.amplitude_m),
        "--frequency_hz", str(args.frequency_hz),
        "--translation_m", str(args.translation_m),
        "--seed", str(args.seed),
        "--progress-every", str(args.progress_every),
        "--imu-noise-seed", str(args.imu_noise_seed),
        "--imu-accel-white-noise-sigma-mps2", str(args.imu_accel_white_noise_sigma_mps2),
        "--imu-gyro-white-noise-sigma-radps", str(args.imu_gyro_white_noise_sigma_radps),
        "--imu-accel-initial-bias-sigma-mps2", str(args.imu_accel_initial_bias_sigma_mps2),
        "--imu-gyro-initial-bias-sigma-radps", str(args.imu_gyro_initial_bias_sigma_radps),
        "--imu-accel-bias-rw-sigma-mps2-sqrt-s", str(args.imu_accel_bias_rw_sigma_mps2_sqrt_s),
        "--imu-gyro-bias-rw-sigma-radps-sqrt-s", str(args.imu_gyro_bias_rw_sigma_radps_sqrt_s),
        "--task", args.task,
        "--learned-checkpoint", str(args.learned_checkpoint.expanduser()),
        "--gate-checkpoint", str(args.gate_checkpoint.expanduser()),
        "--visibility-checkpoint", str(args.visibility_checkpoint.expanduser()),
        "--replay-npz", str(replay_npz),
        "--output-dir", str(output_dir),
        "--device", args.device,
    ]
    if args.learned_fusion_rate_hz is not None:
        command.extend(
            ["--learned-fusion-rate-hz", str(args.learned_fusion_rate_hz)]
        )
    optional_noise_args = (
        ("--ekf-accel-noise-sigma", args.ekf_accel_noise_sigma),
        ("--ekf-gyro-noise-sigma", args.ekf_gyro_noise_sigma),
        ("--ekf-accel-bias-rw-sigma", args.ekf_accel_bias_rw_sigma),
        ("--ekf-gyro-bias-rw-sigma", args.ekf_gyro_bias_rw_sigma),
    )
    for flag, value in optional_noise_args:
        if value is not None:
            command.extend([flag, str(value)])
    if args.headless:
        command.append("--headless")
    if args.disable_visibility:
        command.append("--disable-visibility")
    return command


def main() -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_npz = output_dir / "mode_A_replay.npz"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    summaries = {}
    for mode in ("A", "S", "B", "P", "C"):
        print(
            f"\n[diagnostics] launching {mode}: {MODE_LABELS[mode]}",
            flush=True,
        )
        subprocess.run(
            _worker_command(mode, output_dir, replay_npz),
            check=True,
            env=env,
        )
        summary_path = output_dir / f"mode_{mode}_summary.json"
        summaries[mode] = json.loads(summary_path.read_text(encoding="utf-8"))

    report = {
        "schema": "isaac_drone_racer.learned_inertial_diagnostics.v1",
        "task": args.task,
        "profile": args.profile,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "modes": summaries,
    }
    report_path = output_dir / "diagnostics_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== LEARNED-INERTIAL DIAGNOSTICS ===", flush=True)
    for mode in ("A", "S", "B", "P", "C"):
        s = summaries[mode]
        pred = s["tcn_prediction_error_vs_gt"]
        print(
            f"{mode} {s['label']:27s} "
            f"pos={s['position_rmse_m']:.4f}m "
            f"vel={s['velocity_rmse_mps']:.4f}m/s "
            f"ori={s['orientation_rmse_deg']:.3f}deg "
            f"pred_rmse={pred['norm_rmse_m']} "
            f"pred_hz={s['learned_update_hz_after_warmup']:.2f} "
            f"fuse_hz={s['learned_fusion_hz_after_warmup']:.2f} "
            f"gate={s['gate_accepted']}/{s['gate_attempts']}"
        )

    print(f"\n[diagnostics] summary={report_path}", flush=True)


if __name__ == "__main__":
    main()
