"""Inspect the gate USD actor frame and geometric bounds.

Run this inside the Isaac/pxr Python environment. Bounding boxes are diagnostic;
the script does not pretend the full mesh bounding box equals the gate opening.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from perception.gate_usd_inspector import inspect_gate_usd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", default="assets/gate/gate.usd")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    report = inspect_gate_usd(args.usd)
    text = json.dumps(asdict(report), indent=2)
    print(text)

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
