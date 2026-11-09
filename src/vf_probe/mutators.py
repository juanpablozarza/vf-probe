"""Trace mutators. Each declares a target component and drop/hold expectation."""

from __future__ import annotations

from vf_probe.loader import EnvProfile
from vf_probe.types import Mutator


def default_mutators(profile: EnvProfile) -> list[Mutator]:
    """Answer-span, cross-task swap, tool-use (only if claimed), and hold.

    Tool-use mutators are omitted when the env does not claim to check
    grounding or tool use, so answer-only graders are not flagged for
    ignoring tool outputs.
    """
    raise NotImplementedError
