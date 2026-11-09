"""Load a taskset and record how it scores, without running the probe."""

from __future__ import annotations

from dataclasses import dataclass

from vf_probe.types import AnswerLocator


@dataclass(frozen=True)
class EnvProfile:
    """What the loader can tell from a taskset before scoring.

    The v1 API has no parser. `has_parser` is almost always False; the
    answer locator then falls back to an extract function or the whole
    final reply. Tool-use mutators run only when `claims_grounding` or
    `claims_tool_use` is True. `needs_judge` (~13% of Hub envs) and
    `needs_sandbox` (~25%) are skipped in v1.
    """

    env_id: str
    answer_field: str | None
    has_parser: bool
    weighted_components: tuple[str, ...]
    claims_grounding: bool
    claims_tool_use: bool
    needs_judge: bool
    needs_sandbox: bool


def load(env_id: str) -> EnvProfile:
    """Load tasks and detect answer field, rewards, and skip/claim flags."""
    raise NotImplementedError


def locate_answer(trace: object, task: object) -> tuple[str, AnswerLocator]:
    """Return `(answer_text, method)`.

    Fall back parser -> own extract function -> whole final reply.
    """
    raise NotImplementedError
