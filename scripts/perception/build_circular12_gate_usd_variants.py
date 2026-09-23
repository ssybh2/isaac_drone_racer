"""Generate and validate the 12 color-coded Circular-12 gate USD assets."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Rebuild all 12 USD variants.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

from tasks.drone_racer.gate_texture_variants import (  # noqa: E402
    ensure_circular12_gate_usd_variants,
    validate_circular12_gate_usd_variants,
)


def main() -> None:
    paths = ensure_circular12_gate_usd_variants(force=bool(args.force))
    report = validate_circular12_gate_usd_variants()

    print("=" * 96)
    print("CIRCULAR-12 GATE USD / TEXTURE BINDINGS")
    print("=" * 96)
    for gate_id in map(str, range(1, 13)):
        print(f"Gate {int(gate_id):02d}: {paths[gate_id]}")
        for asset_path in report[gate_id]:
            print(f"           -> {asset_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
