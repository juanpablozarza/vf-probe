"""Bootstrap over tasks and Holm-correct env x mutator tests."""

from __future__ import annotations

from collections.abc import Sequence

from vf_probe.types import Report, Row


def analyze(rows: Sequence[Row]) -> Report:
    """Add bootstrap CIs over tasks (not rollouts) and Holm-correct p-values.

    Flags use total-score sensitivity. Per-component attribution only
    when a row's `attribution` is `"per_component"` (>=2 weighted
    components). Zero-weight metrics stay evidence only.
    """
    raise NotImplementedError
