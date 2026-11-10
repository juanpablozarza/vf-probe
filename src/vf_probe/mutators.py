"""Trace mutators that do not know an environment's reply format.

``swap_answer`` replaces the gold answer span inside the final reply.
``swap_observation`` replaces every tool result and leaves that reply alone.

Which edit runs is decided from the reward, not from its name. An ``evidence``
attribute on the hook wins. Otherwise the hook source, plus one level of
helpers it calls, is scanned for the evidence it reads. A single opaque
reward gets both edits. Several opaque rewards are reported and not flagged.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf

from vf_probe.loader import locate_answer
from vf_probe.support import reward_hooks
from vf_probe.types import Mutator

_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_MISSING = object()
_ANSWER_READS = (
    "last_reply",
    "final_answer",
    "gold_answer",
    "expected_answer",
    ".gold",
    '["gold"]',
    "['gold']",
    '["answer"]',
    "['answer']",
)
_OBSERVATION_READS = ("tool_messages", ".steps")
_SKIP_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "getattr",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)

OBSERVATION_REDACTION = "PROBE_OBSERVATION"
ANSWER_REDACTION = "PROBE_ANSWER"


@dataclass(frozen=True)
class PlannedEdit:
    """One edit of one reward. Unattributed edits are scored and not flagged."""

    mutator: Mutator
    attributed: bool


def declares_evidence(*kinds: str):
    """Mark a reward with the evidence its author says it reads."""

    evidence = frozenset(kinds)

    def mark(fn: Any) -> Any:
        fn.evidence = evidence
        return fn

    return mark


def default_mutators(task: object, reward_names: Sequence[str]) -> list[PlannedEdit]:
    """Edits for the rewards that actually scored."""
    return select_mutators(task, reward_names)


def select_mutators(task: object, reward_names: Sequence[str]) -> list[PlannedEdit]:
    """Choose edits from declared evidence, else from what the function reads."""
    hooks = {getattr(fn, "__name__", ""): fn for fn in reward_hooks(task)}
    names = list(reward_names)
    alone = len(names) == 1
    chosen: list[PlannedEdit] = []
    for name in names:
        kinds, attributed = _kinds_for(hooks.get(name), alone=alone)
        for kind, mutator_name in (
            ("answer", "swap_answer"),
            ("observation", "swap_observation"),
        ):
            if kind not in kinds:
                continue
            chosen.append(
                PlannedEdit(Mutator(mutator_name, name, "drop"), attributed)
            )
    return chosen


def _kinds_for(fn: Any, *, alone: bool) -> tuple[frozenset[str], bool]:
    if fn is not None:
        declared = _declared_evidence(fn)
        if declared is not None:
            return declared, True
        read = evidence_from_source(fn)
        if read:
            return read, True
    if alone:
        return frozenset({"answer", "observation"}), True
    return frozenset({"answer", "observation"}), False


def _declared_evidence(fn: Any) -> frozenset[str] | None:
    raw = getattr(fn, "evidence", _MISSING)
    if raw is _MISSING:
        return None
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, (frozenset, set, tuple, list)):
        return frozenset(str(item) for item in raw)
    return frozenset()


def evidence_from_source(fn: Any) -> frozenset[str]:
    """Evidence named in the hook and in helpers it calls directly."""
    try:
        own = inspect.getsource(fn)
    except (OSError, TypeError):
        return frozenset()
    chunks = [own]
    for name in _called_names(own):
        if name in _SKIP_CALLS or name == getattr(fn, "__name__", ""):
            continue
        helper = _resolve_helper(fn, name)
        if helper is None:
            continue
        try:
            chunks.append(inspect.getsource(helper))
        except (OSError, TypeError):
            continue
    text = "\n".join(chunks)
    found: set[str] = set()
    if any(marker in text for marker in _ANSWER_READS):
        found.add("answer")
    if any(marker in text for marker in _OBSERVATION_READS):
        found.add("observation")
    return frozenset(found)


def _called_names(source: str) -> list[str]:
    return list(dict.fromkeys(_CALL.findall(source)))


def _resolve_helper(fn: Any, name: str) -> Any:
    globs = getattr(fn, "__globals__", {})
    helper = globs.get(name)
    if helper is None:
        helper = _method(fn, name)
    if helper is None or helper is fn or isinstance(helper, type):
        return None
    if not callable(helper):
        return None
    return helper


def _method(fn: Any, name: str) -> Any:
    globs = getattr(fn, "__globals__", {})
    qualname = getattr(fn, "__qualname__", "")
    if "." not in qualname:
        return None
    owner = globs.get(qualname.split(".")[0])
    if not isinstance(owner, type):
        return None
    return owner.__dict__.get(name)


def swap_answer(trace: vf.Trace, task: object) -> vf.Trace | None:
    """Replace the located answer span in the final reply. Tool results stay."""
    span, _method = locate_answer(trace, task)
    if not span.strip():
        return None
    cloned = trace.model_copy(deep=True)
    if not _replace_last_reply(cloned, span, _redaction(span, ANSWER_REDACTION)):
        return None
    return cloned


def swap_observation(trace: vf.Trace, _task: object) -> vf.Trace | None:
    """Replace every tool result. The final reply stays, including its citation."""
    if not trace.tool_messages:
        return None
    cloned = trace.model_copy(deep=True)
    for node in cloned.nodes:
        message = node.message
        if isinstance(message, vf.ToolMessage):
            message.content = OBSERVATION_REDACTION
    return cloned


def apply_mutator(spec: Mutator, trace: vf.Trace, task: object) -> vf.Trace | None:
    if spec.name == "swap_answer":
        return swap_answer(trace, task)
    if spec.name == "swap_observation":
        return swap_observation(trace, task)
    return None


def _redaction(span: str, token: str) -> str:
    redaction = token
    while span in redaction:
        redaction += "_X"
    return redaction


def _replace_last_reply(trace: vf.Trace, span: str, redaction: str) -> bool:
    for node in reversed(trace.nodes):
        message = node.message
        if not node.sampled or not isinstance(message, vf.AssistantMessage):
            continue
        content = message.content
        if content is None or span not in content:
            continue
        message.content = content.replace(span, redaction)
        return True
    return False
