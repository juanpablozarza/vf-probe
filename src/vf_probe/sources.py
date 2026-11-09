"""Trace sources: synthetic degenerates, gold, and (v2) real traces."""

from __future__ import annotations


def synthetic(task: object) -> list[object]:
    """Degenerate replies: empty, prompt echo, yes, refusal, number list."""
    raise NotImplementedError


def gold(task: object) -> list[object]:
    """Oracle traces for the task, if the env exposes one."""
    raise NotImplementedError


def real(env_id: str) -> list[object]:
    """v2: published traces, then cheap-model rollouts under an approved cap."""
    raise NotImplementedError
