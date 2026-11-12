"""Score a trace and the mutations that should move its rewards.

The caller supplies the trace. A gold reply, a published trajectory, or a
rollout all enter the same way. The mutators rewrite tool results and the
answer span. They do not know the environment's format.

A judge, a container, or a reward that requires a runtime is not scored.
A gold score of 0 is not a clean bill.
"""

from __future__ import annotations

from collections.abc import Sequence

import verifiers.v1 as vf

from vf_probe.mutators import answer_edit_traces, apply_mutator, select_mutators
from vf_probe.scorer import score
from vf_probe.support import (
    NEEDS_JUDGE,
    NEEDS_SANDBOX,
    NOT_TEXT_SCORABLE,
    has_judge,
    only_sandbox,
    sandbox_component_message,
    sandbox_reward_names,
)
from vf_probe.types import (
    ComponentSkip,
    Finding,
    Mutator,
    ProbeResult,
    SkipReason,
    Unattributed,
)


async def probe(
    task: vf.Task,
    trace: vf.Trace,
    *,
    eps: float = 0.0,
    other_golds: Sequence[str] = (),
) -> ProbeResult:
    """Probe one trace. `result.clean` is the clean bill, not an empty list."""
    if has_judge(task):
        return _unsupported("needs_judge", NEEDS_JUDGE)
    if only_sandbox(task):
        return _unsupported(
            "needs_sandbox",
            NEEDS_SANDBOX,
            _sandbox_skips(sandbox_reward_names(task)),
        )

    blocked = set(sandbox_reward_names(task))
    base = await score(task, trace, runtime=None)
    names = [name for name in _weighted_names(base) if name not in blocked]
    plans = select_mutators(task, names)
    findings: list[Finding] = []
    unattributed: list[Unattributed] = []
    checked = 0
    for plan in plans:
        spec = plan.mutator
        base_score = _weighted_score(base, spec.target_component)
        if base_score is None or base_score <= 0:
            continue
        for edit_name, mutated in _edits(spec, trace, task, other_golds):
            if mutated is None:
                continue
            mutant = await score(task, mutated, runtime=None)
            mutant_score = _weighted_score(mutant, spec.target_component)
            if mutant_score is None:
                continue
            if not plan.attributed:
                unattributed.append(
                    Unattributed(
                        mutator=edit_name,
                        component=spec.target_component,
                        base=base_score,
                        mutant=mutant_score,
                    )
                )
                continue
            checked += 1
            if _failed(spec, base_score, mutant_score, eps):
                findings.append(
                    Finding(
                        mutator=edit_name,
                        component=spec.target_component,
                        base=base_score,
                        mutant=mutant_score,
                    )
                )
    skips = _sandbox_skips(sorted(blocked))
    noted = tuple(unattributed)
    if checked == 0 and not noted:
        return _unsupported("not_text_scorable", NOT_TEXT_SCORABLE, skips)
    return ProbeResult(
        findings=tuple(findings),
        reason=None,
        message=None,
        clean=checked > 0 and not findings,
        skips=skips,
        unattributed=noted,
    )


def _edits(
    spec: Mutator,
    trace: vf.Trace,
    task: vf.Task,
    other_golds: Sequence[str],
) -> list[tuple[str, vf.Trace]]:
    if spec.name == "swap_answer":
        return answer_edit_traces(trace, task, other_golds)
    mutated = apply_mutator(spec, trace, task)
    if mutated is None:
        return []
    return [(spec.name, mutated)]


def _unsupported(
    reason: SkipReason,
    message: str,
    skips: tuple[ComponentSkip, ...] = (),
) -> ProbeResult:
    return ProbeResult(
        findings=(),
        reason=reason,
        message=message,
        clean=False,
        skips=skips,
    )


def _sandbox_skips(names: list[str]) -> tuple[ComponentSkip, ...]:
    return tuple(
        ComponentSkip(
            component=name,
            reason="needs_sandbox",
            message=sandbox_component_message(name),
        )
        for name in names
    )


def _weighted_names(trace: vf.Trace) -> list[str]:
    names: list[str] = []
    for name, reward in trace.rewards.items():
        if reward is not None and reward.weight > 0:
            names.append(name)
    return names


def _weighted_score(trace: vf.Trace, name: str) -> float | None:
    reward = trace.rewards.get(name)
    if reward is None or reward.weight <= 0:
        return None
    return reward.score


def _failed(spec: Mutator, base: float, mutant: float, eps: float) -> bool:
    if spec.expect != "drop" or base <= 0:
        return False
    return mutant + eps >= base
