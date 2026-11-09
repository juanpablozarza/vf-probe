"""Shared types: Mutator, Row, Report."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

Expect = Literal["drop", "hold"]
AnswerLocator = Literal["parser", "extract", "final_reply"]
Attribution = Literal["total", "per_component"]
SkipReason = Literal["needs_judge", "needs_sandbox", "not_text_scorable"]


@dataclass(frozen=True)
class Mutator:
    """One probe mutation.

    `target_component` is a named weighted reward, or `"reward"` for the
    total score. v1 usually targets `"reward"` because few Hub envs expose
    two or more weighted components.
    """

    name: str
    target_component: str
    expect: Expect

    def apply(self, trace: object, task: object, rng: Random) -> object | None:
        """Return a mutated trace, or None if this mutator does not apply."""
        raise NotImplementedError


@dataclass(frozen=True)
class Row:
    """One env x mutator result."""

    env_id: str
    mutator: str
    verifiers_version: str
    answer_locator: AnswerLocator
    attribution: Attribution
    k: int
    n: int
    skipped: SkipReason | None = None


@dataclass(frozen=True)
class Report:
    """JSON/Markdown payload: one row per env x mutator, plus fixture paths."""

    rows: tuple[Row, ...]
    fixtures: tuple[str, ...] = ()
