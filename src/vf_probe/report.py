"""Write JSON, Markdown, and one fixture per flag."""

from __future__ import annotations

from pathlib import Path

from vf_probe.types import Report


def write(report: Report, out: Path) -> None:
    """Write JSON, a Markdown table, and one fixture (trace + scores) per flag."""
    raise NotImplementedError
