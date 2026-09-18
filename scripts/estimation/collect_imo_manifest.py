"""Collect a complete IMO dataset from an explicit V2 manifest.

Each trajectory is launched in a fresh Isaac process. The same manifest also
contains the train/val/test split consumed by train_learned_motion.py, so split
membership is fixed before collection and cannot accidentally leak a motion
family into training only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument(
    "--only-split",
    choices=("train", "val", "test"),
    default=None,
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip traces whose CSV and sidecar JSON already exist.",
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


_ALLOWED_COLLECTOR_KEYS = {
    "steps",
    "profile",
    "amplitude_m",
    "frequency_hz",
    "translation_m",
    "vertical_m",
    "motion_duration_s",
    "yaw_amplitude_deg",
    "yaw_frequency_hz",
    "phase_rad",
    "vehicle_mass_kg",
    "seed",
    "task",
}


def _load_manifest():
    path = args.manifest.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "isaac_drone_racer.imo_dataset_manifest.v2":
        raise ValueError(f"unsupported manifest schema: {data.get('schema')!r}")
    path_base = Path(data.get("path_base", "."))
    base_dir = (
        path_base
        if path_base.is_absolute()
        else (path.parent / path_base).resolve()
    )
    defaults = dict(data.get("collector_defaults", {}))
    unknown_defaults = set(defaults) - _ALLOWED_COLLECTOR_KEYS
    if unknown_defaults:
        raise ValueError(f"unknown collector default keys: {sorted(unknown_defaults)}")
    traces = data.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("manifest must contain a non-empty traces list")
    return path, base_dir, defaults, traces


def _output_path(base_dir: Path, entry: dict) -> Path:
    raw = entry.get("path")
    if not raw:
        raise ValueError("each trace entry must define path")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _command(
    collector: Path,
    output: Path,
    split: str,
    params: dict,
) -> list[str]:
    command = [
        sys.executable,
        str(collector),
        "--output",
        str(output),
        "--dataset_split",
        split,
        "--device",
        args.device,
    ]
    if args.headless:
        command.append("--headless")

    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def main() -> None:
    manifest_path, base_dir, defaults, traces = _load_manifest()
    collector = Path(__file__).with_name("collect_imo_supervised_data.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    selected = []
    for entry in traces:
        if not isinstance(entry, dict):
            raise ValueError("every manifest trace entry must be an object")
        split = entry.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError(f"invalid trace split: {split!r}")
        if args.only_split is not None and split != args.only_split:
            continue
        selected.append(entry)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]

    print(
        f"[IMO manifest] manifest={manifest_path} traces={len(selected)} "
        f"headless={args.headless} resume={args.resume}",
        flush=True,
    )

    completed = 0
    skipped = 0
    for index, entry in enumerate(selected, start=1):
        split = entry["split"]
        output = _output_path(base_dir, entry)
        sidecar = output.with_suffix(output.suffix + ".json")
        name = entry.get("name", output.stem)

        params = dict(defaults)
        params.update(entry.get("collector", {}))
        unknown = set(params) - _ALLOWED_COLLECTOR_KEYS
        if unknown:
            raise ValueError(
                f"trace {name!r} has unknown collector keys: {sorted(unknown)}"
            )

        if args.resume and output.exists() and sidecar.exists():
            print(
                f"[IMO manifest] {index}/{len(selected)} skip existing "
                f"{split}:{name}",
                flush=True,
            )
            skipped += 1
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        command = _command(collector, output, split, params)
        print(
            f"\n[IMO manifest] {index}/{len(selected)} collect {split}:{name}",
            flush=True,
        )
        print("[IMO manifest] " + " ".join(command), flush=True)

        if not args.dry_run:
            subprocess.run(command, check=True, env=env)
            completed += 1

    print(
        f"\n[IMO manifest] done completed={completed} skipped={skipped} "
        f"selected={len(selected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
