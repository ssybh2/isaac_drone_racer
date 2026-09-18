"""Sweep learned-motion EKF update rates on one fixed replay trajectory.

This isolates correlation from model quality. The learned window remains 0.5 s
while the fusion cadence changes. At 20 Hz adjacent windows overlap by 90%;
at 10 Hz by 80%; at 5 Hz by 60%; at 2 Hz they are non-overlapping.

Mode A replay must already exist (for example from diagnose_learned_inertial.py).
Each B-mode rate runs in a fresh Isaac process to avoid SimulationApp lifecycle
contamination.
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
parser.add_argument(
    "--rates-hz",
    type=float,
    nargs="+",
    default=[20.0, 10.0, 5.0, 2.0],
)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--profile", choices=("hover", "translate_x", "circle", "lissajous"), default="translate_x")
parser.add_argument("--amplitude_m", type=float, default=1.5)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--progress-every", type=int, default=100)
parser.add_argument(
    "--task",
    default="Isaac-Drone-Racer-Learned-Inertial-v0",
)
parser.add_argument("--learned-checkpoint", type=Path, required=True)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/learned_inertial_diagnostics/update_rate_sweep"),
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


def _worker_command(rate_hz: float, output_dir: Path) -> list[str]:
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
        "--learned-update-rate-hz", str(rate_hz),
        "--replay-npz", str(args.replay_npz.expanduser()),
        "--output-dir", str(output_dir),
        "--device", args.device,
    ]
    if args.headless:
        cmd.append("--headless")
    return cmd


def _rate_tag(rate_hz: float) -> str:
    return ("%g" % rate_hz).replace(".", "p")


def main() -> None:
    replay = args.replay_npz.expanduser().resolve()
    if not replay.exists():
        raise FileNotFoundError(f"replay not found: {replay}")
    checkpoint = args.learned_checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    results = {}
    for rate_hz in args.rates_hz:
        rate = float(rate_hz)
        if rate <= 0.0:
            raise ValueError("all --rates-hz values must be positive")
        out = root / f"{_rate_tag(rate)}hz"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[rate-sweep] launching B at {rate:g} Hz", flush=True)
        subprocess.run(
            _worker_command(rate, out),
            check=True,
            env=env,
        )
        summary = json.loads(
            (out / "mode_B_summary.json").read_text(encoding="utf-8")
        )
        results[str(rate)] = summary

    report = {
        "schema": "isaac_drone_racer.learned_update_rate_sweep.v1",
        "window_time_s": 0.5,
        "rates_hz": [float(v) for v in args.rates_hz],
        "replay_npz": str(replay),
        "checkpoint": str(checkpoint),
        "results": results,
    }
    path = root / "rate_sweep_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== LEARNED UPDATE RATE SWEEP ===")
    for rate_hz in args.rates_hz:
        s = results[str(float(rate_hz))]
        pred = s["tcn_prediction_error_vs_gt"]
        overlap = max(0.0, 1.0 - 1.0 / (float(rate_hz) * 0.5))
        print(
            f"{float(rate_hz):5.1f} Hz "
            f"overlap={100.0 * overlap:5.1f}% "
            f"pos={s['position_rmse_m']:.4f}m "
            f"vel={s['velocity_rmse_mps']:.4f}m/s "
            f"ori={s['orientation_rmse_deg']:.3f}deg "
            f"pred={pred['norm_rmse_m']:.6f}m "
            f"updates={s['learned_updates']}"
        )
    print(f"\n[rate-sweep] summary={path}")


if __name__ == "__main__":
    main()
