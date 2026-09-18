"""Sweep learned-measurement covariance multipliers at a fixed update rate.

The 0.5 s learned-motion windows are strongly correlated when fused faster than
2 Hz. This tool keeps the update cadence fixed (20 Hz by default) and varies
only the final EKF measurement covariance multiplier. It tests whether the
observed degradation is primarily information over-counting rather than a
problem in the learned residual itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay-npz", type=Path, required=True)
parser.add_argument("--learned-checkpoint", type=Path, required=True)
parser.add_argument(
    "--multipliers",
    type=float,
    nargs="+",
    default=[1.0, 2.0, 5.0, 10.0, 20.0],
)
parser.add_argument("--learned-update-rate-hz", type=float, default=20.0)
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
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-v0",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(
        "artifacts/learned_inertial_diagnostics/covariance_multiplier_sweep"
    ),
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


def _tag(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def _worker_command(multiplier: float, output_dir: Path) -> list[str]:
    worker = Path(__file__).with_name("evaluate_learned_inertial_mode.py")
    cmd = [
        sys.executable,
        str(worker),
        "--mode", "B",
        "--steps", str(args.steps),
        "--profile", args.profile,
        "--amplitude_m", str(args.amplitude_m),
        "--frequency_hz", str(args.frequency_hz),
        "--translation_m", str(args.translation_m),
        "--seed", str(args.seed),
        "--progress-every", str(args.progress_every),
        "--task", args.task,
        "--learned-checkpoint", str(args.learned_checkpoint.expanduser()),
        "--learned-update-rate-hz", str(args.learned_update_rate_hz),
        "--learned-covariance-multiplier", str(multiplier),
        "--replay-npz", str(args.replay_npz.expanduser()),
        "--output-dir", str(output_dir),
        "--device", args.device,
    ]
    if args.headless:
        cmd.append("--headless")
    return cmd


def main() -> None:
    replay = args.replay_npz.expanduser().resolve()
    if not replay.exists():
        raise FileNotFoundError(f"replay not found: {replay}")
    checkpoint = args.learned_checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.learned_update_rate_hz <= 0.0:
        raise ValueError("--learned-update-rate-hz must be positive")

    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    results = {}
    for value in args.multipliers:
        multiplier = float(value)
        if multiplier <= 0.0:
            raise ValueError("all covariance multipliers must be positive")
        out = root / f"x{_tag(multiplier)}"
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[cov-sweep] launching B at "
            f"{args.learned_update_rate_hz:g} Hz with covariance x{multiplier:g}",
            flush=True,
        )
        subprocess.run(
            _worker_command(multiplier, out),
            check=True,
            env=env,
        )
        summary = json.loads(
            (out / "mode_B_summary.json").read_text(encoding="utf-8")
        )
        results[str(multiplier)] = summary

    report = {
        "schema": "isaac_drone_racer.learned_covariance_multiplier_sweep.v1",
        "learned_update_rate_hz": float(args.learned_update_rate_hz),
        "window_time_s": 0.5,
        "multipliers": [float(v) for v in args.multipliers],
        "replay_npz": str(replay),
        "checkpoint": str(checkpoint),
        "results": results,
    }
    path = root / "covariance_sweep_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== LEARNED COVARIANCE MULTIPLIER SWEEP ===")
    for value in args.multipliers:
        multiplier = float(value)
        s = results[str(multiplier)]
        pred = s["tcn_prediction_error_vs_gt"]
        print(
            f"x{multiplier:5.1f} "
            f"pos={s['position_rmse_m']:.4f}m "
            f"vel={s['velocity_rmse_mps']:.4f}m/s "
            f"ori={s['orientation_rmse_deg']:.3f}deg "
            f"pred={pred['norm_rmse_m']:.6f}m "
            f"updates={s['learned_updates']}"
        )
    print(f"\n[cov-sweep] summary={path}")


if __name__ == "__main__":
    main()
