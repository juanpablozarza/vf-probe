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
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf

from vf_probe.answer_edits import answer_replacements, gold_kind
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
_TOOL_CALL_READS = ("tool_calls",)
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
            ("tool_call", "rename_tool"),
            ("tool_call", "perturb_argument"),
            ("tool_call", "swap_arguments"),
            ("tool_call", "delete_tool_call"),
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
    if any(marker in text for marker in _TOOL_CALL_READS):
        found.add("tool_call")
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


def swap_answer(
    trace: vf.Trace,
    task: object,
    other_golds: Sequence[str] = (),
) -> vf.Trace | None:
    """Apply the answer edit that matches the span. Tool results stay."""
    edits = _answer_edits(trace, task, other_golds)
    if not edits:
        return None
    _name, replacement, old = edits[0]
    return _replace_answer(trace, old, replacement)


def answer_edit_traces(
    trace: vf.Trace,
    task: object,
    other_golds: Sequence[str] = (),
) -> list[tuple[str, vf.Trace]]:
    """Every applicable answer edit, each on its own copy of `trace`."""
    edited: list[tuple[str, vf.Trace]] = []
    for name, replacement, old in _answer_edits(trace, task, other_golds):
        mutated = _replace_answer(trace, old, replacement)
        if mutated is not None:
            edited.append((name, mutated))
    return edited


def _answer_edits(
    trace: vf.Trace,
    task: object,
    other_golds: Sequence[str],
) -> list[tuple[str, str, str]]:
    old = _graded_answer(trace, task)
    if not old.strip():
        return []
    return [
        (name, replacement, old)
        for name, replacement in answer_replacements(
            old,
            gold_kind=gold_kind(task),
            other_golds=other_golds,
        )
    ]


def _graded_answer(trace: vf.Trace, task: object) -> str:
    final = _final_answer(trace)
    if isinstance(final, dict):
        answer = final.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
    span, _method = locate_answer(trace, task)
    return span


def _final_answer(trace: vf.Trace) -> object:
    return getattr(getattr(trace, "state", None), "final_answer", None)


def _replace_answer(trace: vf.Trace, old: str, new: str) -> vf.Trace | None:
    cloned = trace.model_copy(deep=True)
    changed = _replace_last_reply(cloned, old, new)
    changed = _replace_final_answer(cloned, old, new) or changed
    return cloned if changed else None


def _replace_final_answer(trace: vf.Trace, old: str, new: str) -> bool:
    state = getattr(trace, "state", None)
    if state is None:
        return False
    final = getattr(state, "final_answer", None)
    if not isinstance(final, dict):
        return False
    answer = final.get("answer")
    if isinstance(answer, str) and old in answer:
        updated = dict(final)
        updated["answer"] = answer.replace(old, new)
    elif isinstance(answer, bool):
        updated = dict(final)
        updated["answer"] = not answer
    else:
        return False
    if hasattr(state, "model_copy"):
        trace.state = state.model_copy(update={"final_answer": updated})
    else:
        setattr(state, "final_answer", updated)
    return True


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
    if spec.name == "rename_tool":
        return rename_tool(trace)
    if spec.name == "perturb_argument":
        return perturb_argument(trace)
    if spec.name == "swap_arguments":
        return swap_arguments(trace)
    if spec.name == "delete_tool_call":
        return delete_tool_call(trace)
    return None


def rename_tool(trace: vf.Trace) -> vf.Trace | None:
    """Point one tool call at a different name. Arguments stay."""
    sites = _call_sites(trace)
    if not sites:
        return None
    site = _cited_site(trace, sites) or sites[0]
    cloned = trace.model_copy(deep=True)
    call = _call_by_id(cloned, site.call_id)
    if call is None:
        return None
    call.name = _other_name(site.name, sites)
    _rename_step(cloned, site.name, site.arguments, call.name)
    return cloned


def perturb_argument(trace: vf.Trace) -> vf.Trace | None:
    """Change one argument value. The function name and the tool result stay."""
    sites = _call_sites(trace)
    preferred = _cited_site(trace, sites)
    ordered = [preferred, *sites] if preferred is not None else sites
    for site in ordered:
        if site is None:
            continue
        key = _perturbable_key(site.arguments)
        if key is None:
            continue
        cloned = trace.model_copy(deep=True)
        call = _call_by_id(cloned, site.call_id)
        if call is None:
            return None
        arguments = json.loads(call.arguments or "{}")
        arguments[key] = _perturb_value(arguments[key])
        call.arguments = json.dumps(arguments)
        _rewrite_step_arguments(cloned, site.name, site.arguments, arguments)
        return cloned
    return None


def swap_arguments(trace: vf.Trace) -> vf.Trace | None:
    """Trade the first two argument values when they differ."""
    sites = _call_sites(trace)
    preferred = _cited_site(trace, sites)
    ordered = [preferred, *sites] if preferred is not None else sites
    for site in ordered:
        if site is None:
            continue
        keys = list(site.arguments)
        if len(keys) < 2 or site.arguments[keys[0]] == site.arguments[keys[1]]:
            continue
        cloned = trace.model_copy(deep=True)
        call = _call_by_id(cloned, site.call_id)
        if call is None:
            return None
        arguments = json.loads(call.arguments or "{}")
        left, right = keys[0], keys[1]
        arguments[left], arguments[right] = arguments[right], arguments[left]
        call.arguments = json.dumps(arguments)
        _rewrite_step_arguments(cloned, site.name, site.arguments, arguments)
        return cloned
    return None


def delete_tool_call(trace: vf.Trace) -> vf.Trace | None:
    """Remove one tool call, its result, and the matching step."""
    sites = _call_sites(trace)
    if not sites:
        return None
    site = _cited_site(trace, sites) or sites[0]
    cloned = trace.model_copy(deep=True)
    for node in cloned.nodes:
        message = node.message
        if isinstance(message, vf.AssistantMessage) and message.tool_calls:
            kept = [call for call in message.tool_calls if call.id != site.call_id]
            message.tool_calls = kept or None
    cloned.nodes = [
        node
        for node in cloned.nodes
        if not (
            isinstance(node.message, vf.ToolMessage)
            and node.message.tool_call_id == site.call_id
        )
    ]
    _drop_step(cloned, site.name, site.arguments)
    return cloned


@dataclass(frozen=True)
class _CallSite:
    call_id: str
    name: str
    arguments: dict[str, Any]


def _call_sites(trace: vf.Trace) -> list[_CallSite]:
    sites: list[_CallSite] = []
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, vf.AssistantMessage):
            continue
        for call in message.tool_calls or []:
            sites.append(
                _CallSite(call.id, call.name, _load_arguments(call.arguments))
            )
    return sites


def _load_arguments(raw: str | None) -> dict[str, Any]:
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _call_by_id(trace: vf.Trace, call_id: str) -> vf.ToolCall | None:
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, vf.AssistantMessage):
            continue
        for call in message.tool_calls or []:
            if call.id == call_id:
                return call
    return None


def _other_name(name: str, sites: list[_CallSite]) -> str:
    used = {site.name for site in sites}
    candidate = f"{name}_other"
    while candidate in used:
        candidate += "_x"
    return candidate


def _perturbable_key(arguments: dict[str, Any]) -> str | None:
    for key, value in arguments.items():
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return key
    return None


def _perturb_value(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + max(1, abs(value))
    if isinstance(value, str):
        flipped = value[::-1]
        if flipped != value and value not in flipped:
            return flipped
        if value not in "other":
            return "other"
        return "other2"
    return value


def _steps(trace: vf.Trace) -> list[Any] | None:
    state = getattr(trace, "state", None)
    steps = getattr(state, "steps", None)
    return steps if isinstance(steps, list) else None


def _matching_step(
    trace: vf.Trace, name: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    steps = _steps(trace)
    if steps is None:
        return None
    for step in steps:
        if (
            isinstance(step, dict)
            and step.get("tool") == name
            and step.get("arguments") == arguments
        ):
            return step
    return None


def _rename_step(
    trace: vf.Trace, name: str, arguments: dict[str, Any], new_name: str
) -> None:
    step = _matching_step(trace, name, arguments)
    if step is not None:
        step["tool"] = new_name


def _rewrite_step_arguments(
    trace: vf.Trace,
    name: str,
    old: dict[str, Any],
    new: dict[str, Any],
) -> None:
    step = _matching_step(trace, name, old)
    if step is not None:
        step["arguments"] = dict(new)


def _drop_step(trace: vf.Trace, name: str, arguments: dict[str, Any]) -> None:
    steps = _steps(trace)
    if steps is None:
        return
    for index, step in enumerate(steps):
        if (
            isinstance(step, dict)
            and step.get("tool") == name
            and step.get("arguments") == arguments
        ):
            del steps[index]
            return


def _cited_site(trace: vf.Trace, sites: list[_CallSite]) -> _CallSite | None:
    ids = _citation_ids(trace)
    if not ids:
        return None
    for site in sites:
        for node in trace.nodes:
            message = node.message
            if not isinstance(message, vf.ToolMessage):
                continue
            if message.tool_call_id != site.call_id:
                continue
            text = message.content if isinstance(message.content, str) else ""
            if any(cid in text for cid in ids):
                return site
    return None


def _citation_ids(trace: vf.Trace) -> list[str]:
    final = getattr(getattr(trace, "state", None), "final_answer", None)
    if not isinstance(final, dict):
        return []
    citations = final.get("citations")
    if not isinstance(citations, list):
        return []
    ids: list[str] = []
    for citation in citations:
        if isinstance(citation, dict) and isinstance(citation.get("id"), str):
            ids.append(citation["id"])
    return ids


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
