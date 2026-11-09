"""Trace mutators that do not know an environment's reply format.

``swap_answer`` replaces the gold answer span inside the final reply.
``swap_observation`` replaces every tool result and leaves that reply alone.
A tool result is targeted when a weighted reward name refers to grounding,
citation, or tool use, or when the task claims tool use and it has one
weighted reward. Answer-only rewards are left alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import verifiers.v1 as vf

from vf_probe.loader import EnvProfile, locate_answer
from vf_probe.types import Mutator

OBSERVATION_REDACTION = "PROBE_OBSERVATION"
ANSWER_REDACTION = "PROBE_ANSWER"


def default_mutators(profile: EnvProfile) -> list[Mutator]:
    """Declarations for one env, from reward names and tool-use claims."""
    return select_mutators(
        profile.weighted_components,
        claims_tool_use=profile.claims_tool_use or profile.claims_grounding,
    )


def select_mutators(
    reward_names: Sequence[str],
    *,
    claims_tool_use: bool,
) -> list[Mutator]:
    chosen: list[Mutator] = []
    for name in answer_targets(reward_names):
        chosen.append(Mutator("swap_answer", name, "drop"))
    for name in observation_targets(reward_names, claims=claims_tool_use):
        chosen.append(Mutator("swap_observation", name, "drop"))
    return chosen


def answer_targets(reward_names: Sequence[str]) -> list[str]:
    named = [name for name in reward_names if _answerish(name)]
    if named:
        return named
    if len(reward_names) == 1:
        return list(reward_names)
    return []


def observation_targets(reward_names: Sequence[str], *, claims: bool) -> list[str]:
    named = [name for name in reward_names if _toolish(name)]
    if named:
        return named
    if claims and len(reward_names) == 1:
        return list(reward_names)
    return []


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


def metric_claims_tool_use(names: Sequence[str]) -> bool:
    """True when a recorded metric name refers to grounding, citation, or tools."""
    return any(_toolish(name) for name in names)


def apply_mutator(spec: Mutator, trace: vf.Trace, task: object) -> vf.Trace | None:
    if spec.name == "swap_answer":
        return swap_answer(trace, task)
    if spec.name == "swap_observation":
        return swap_observation(trace, task)
    return None


def _answerish(name: str) -> bool:
    lowered = name.lower()
    return lowered in {
        "answer",
        "correct",
        "correctness",
        "exact",
        "match",
    } or lowered.endswith("_answer")


def _toolish(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in ("citat", "ground", "observ", "tool"))


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
