"""CLI stub: `vf-probe <env-id> [--out PATH]`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vf-probe",
        description=(
            "Probe a verifiers environment for graders that reward the wrong thing."
        ),
    )
    parser.add_argument("env_id", help="Environment id (Hub owner/name)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for the JSON report, Markdown table, and fixtures",
    )
    parser.parse_args(argv)
    print("not implemented yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
