"""CLI: regenerate the decision packet from the register (records no decision)."""
from __future__ import annotations

import argparse
import sys

from src.errors import PrototypeError
from src.report.decision_packet import (packet_is_fresh, write_packet)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the decision packet")
    parser.add_argument("--register", default="config/replaceability.yaml")
    parser.add_argument("--out", default="docs/DECISION_PACKET.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        from pathlib import Path

        if args.check:
            fresh = packet_is_fresh(Path(args.register), Path(args.out))
            print("decision packet: fresh" if fresh else
                  "decision packet: stale or missing (run python -m codes.decision_packet)")
            return 0 if fresh else 1
        path = write_packet(Path(args.register), Path(args.out))
    except (OSError, PrototypeError) as exc:
        print(f"decision packet failed: {exc}", file=sys.stderr)
        return 2
    print(f"decision packet written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())