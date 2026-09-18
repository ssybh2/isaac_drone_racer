"""Unified A/B/C learned-inertial estimator evaluation orchestrator.

Each estimator mode runs in a fresh Python/Isaac process. This avoids reusing a
single SimulationApp across repeated gym.make()/env.close() cycles, which can
stall while constructing the second Isaac Lab environment.

A: IMU propagation only
B: IMU + 20 Hz overlapping TCN displacement updates
C: IMU + 20 Hz TCN + Stage2 mapped-gate PnP absolute updates

Mode A generates the action replay once. Modes B and C replay the exact same
actions in fresh processes. Per-mode traces/summaries are merged into the final
CSV/JSON after all three workers complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument(
    "--profile",
    choices=("hover", "translate_x", "circle", "lissajous"),
    default="lissajous",
)
parser.add_argument("--amplitude_m", type=float, default=1.5)
parser.add_argument("--frequency_hz", type=float, default=0.10)
parser.add_argument("--translation_m", type=float, default=6.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--progress-every", type=int, default=100)
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
    default=Path("artifacts/learned_inertial_ab"),
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


MODE_LABELS = {
    "A": "imu_only",
    "B": "imu_tcn_20hz",
    "C": "imu_tcn_gate",
}


def _worker_command(mode: str, output_dir: Path, replay_npz: Path) -> list[str]:
    worker = Path(__file__).with_name("evaluate_learned_inertial_mode.py")
    command = [
        sys.executable,
        str(worker),
        "--mode",
        mode,
        "--steps",
        str(args.steps),
        "--profile",
        args.profile,
        "--amplitude_m",
        str(args.amplitude_m),
        "--frequency_hz",
        str(args.frequency_hz),
        "--translation_m",
        str(args.translation_m),
        "--seed",
        str(args.seed),
        "--progress-every",
        str(args.progress_every),
        "--task",
        args.task,
        "--learned-checkpoint",
        str(args.learned_checkpoint.expanduser()),
        "--gate-checkpoint",
        str(args.gate_checkpoint.expanduser()),
        "--visibility-checkpoint",
        str(args.visibility_checkpoint.expanduser()),
        "--replay-npz",
        str(replay_npz),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    if args.headless:
        command.append("--headless")
    if args.disable_visibility:
        command.append("--disable-visibility")
    return command


def _merge_traces(output_dir: Path) -> Path:
    combined = output_dir / "estimator_ab_trace.csv"
    wrote_header = False
    with combined.open("w", newline="", encoding="utf-8") as dst:
        writer = None
        for mode in ("A", "B", "C"):
            source = output_dir / f"mode_{mode}_trace.csv"
            with source.open("r", newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                if not wrote_header:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
    return combined


def _load_summaries(output_dir: Path) -> dict[str, dict]:
    summaries = {}
    for mode in ("A", "B", "C"):
        path = output_dir / f"mode_{mode}_summary.json"
        summaries[mode] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def main() -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_npz = output_dir / "mode_A_replay.npz"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for mode in ("A", "B", "C"):
        print(
            f"\n[estimator-ab] launching mode {mode}: {MODE_LABELS[mode]} "
            f"in a fresh Isaac process",
            flush=True,
        )
        subprocess.run(
            _worker_command(mode, output_dir, replay_npz),
            check=True,
            env=env,
        )
        print(f"[estimator-ab] mode {mode} process exited cleanly", flush=True)

    summaries = _load_summaries(output_dir)
    combined_trace = _merge_traces(output_dir)
    json_path = output_dir / "estimator_ab_summary.json"

    report = {
        "schema": "isaac_drone_racer.learned_inertial_ab.v2",
        "task": args.task,
        "profile": args.profile,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "execution_contract": (
            "Modes A/B/C run in separate fresh Isaac processes. Mode A writes "
            "the action replay; Modes B/C replay those exact actions. GT is used "
            "only for evaluation and the Mode-A trajectory generator, never as "
            "an estimator update after the fixed known start."
        ),
        "checkpoints": {
            "learned_motion": str(args.learned_checkpoint.expanduser().resolve()),
            "gate_detector": str(args.gate_checkpoint.expanduser().resolve()),
            "visibility": (
                None
                if args.disable_visibility
                else str(args.visibility_checkpoint.expanduser().resolve())
            ),
        },
        "modes": summaries,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n[estimator-ab] wrote combined trace: {combined_trace}", flush=True)
    print(f"[estimator-ab] wrote combined summary: {json_path}", flush=True)
    print("\n=== A/B/C SUMMARY ===", flush=True)
    for mode in ("A", "B", "C"):
        s = summaries[mode]
        print(
            f"{mode} {s['label']:16s} "
            f"pos={s['position_rmse_m']:.4f}m "
            f"vel={s['velocity_rmse_mps']:.4f}m/s "
            f"ori={s['orientation_rmse_deg']:.3f}deg "
            f"max_pos={s['position_max_error_m']:.4f}m "
            f"learned_hz={s['learned_update_hz_after_warmup']:.2f} "
            f"clones_max={s['clone_count_max']} "
            f"skips={s['learned_update_skips']} "
            f"gate={s['gate_accepted']}/{s['gate_attempts']} "
            f"rejected={s['gate_rejected']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
