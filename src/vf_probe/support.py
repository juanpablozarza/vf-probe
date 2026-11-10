"""Decide when a task cannot be probed from its text.

The decision uses the reward signature and ``NEEDS_CONTAINER``. Hub tags are
not consulted.
"""

from __future__ import annotations

import inspect
from typing import Any

NEEDS_SANDBOX = (
    "This environment needs a sandbox. A reward requires a runtime or "
    "deployment, or the task sets NEEDS_CONTAINER. The probe does not call "
    "that reward."
)
NEEDS_JUDGE = "This environment uses a judge. The probe does not call score."
NOT_TEXT_SCORABLE = (
    "This environment produced no text score. Every weighted reward was "
    "missing, or the gold score was 0 on every reward the probe meant to test."
)


def sandbox_component_message(name: str) -> str:
    return (
        f"Reward {name} needs a sandbox. It requires a runtime or deployment, "
        "so the probe does not call it."
    )


def has_judge(task: object) -> bool:
    """True when the task config has a plugged judge. Does not load the judge."""
    config = getattr(task, "config", None)
    return bool(getattr(config, "judges", None))


def needs_container(task: object) -> bool:
    return bool(getattr(type(task), "NEEDS_CONTAINER", False))


def requires_runtime(fn: Any) -> bool:
    """True when `runtime` or `deployment` is a required parameter."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    for name in ("runtime", "deployment"):
        param = parameters.get(name)
        if param is not None and param.default is inspect.Parameter.empty:
            return True
    return False


def reward_hooks(task: object) -> list[Any]:
    hooks = getattr(task, "hooks", None)
    if not callable(hooks):
        return []
    found = hooks("reward")
    return list(found) if isinstance(found, list) else []


def sandbox_reward_names(task: object) -> list[str]:
    names: list[str] = []
    for fn in reward_hooks(task):
        if requires_runtime(fn):
            names.append(getattr(fn, "__name__", "reward"))
    return names


def only_sandbox(task: object) -> bool:
    """The task needs a container, or every reward requires a runtime."""
    if needs_container(task):
        return True
    hooks = reward_hooks(task)
    return bool(hooks) and all(requires_runtime(fn) for fn in hooks)
