"""Score a trace and the mutations that should move its rewards.

The caller supplies the trace. A gold reply, a published trajectory, or a
rollout all enter the same way. The mutators rewrite tool results and the
answer span. They do not know the environment's format.
"""

from __future__ import annotations

import verifiers.v1 as vf

from vf_probe.mutators import apply_mutator, metric_claims_tool_use, select_mutators
from vf_probe.scorer import score
from vf_probe.types import Finding, Mutator


async def probe(
    task: vf.Task,
    trace: vf.Trace,
    *,
    eps: float = 0.0,
) -> list[Finding]:
    """Return rewards that failed to drop. An empty list is a clean grader."""
    base = await score(task, trace, runtime=None)
    names = _weighted_names(base)
    specs = select_mutators(names, claims_tool_use=_claims_tool_use(task, base))
    findings: list[Finding] = []
    for spec in specs:
        mutated = apply_mutator(spec, trace, task)
        if mutated is None:
            continue
        mutant = await score(task, mutated, runtime=None)
        base_score = _weighted_score(base, spec.target_component)
        mutant_score = _weighted_score(mutant, spec.target_component)
        if base_score is None or mutant_score is None:
            continue
        if _failed(spec, base_score, mutant_score, eps):
            findings.append(
                Finding(
                    mutator=spec.name,
                    component=spec.target_component,
                    base=base_score,
                    mutant=mutant_score,
                )
            )
    return findings


def _claims_tool_use(task: object, trace: vf.Trace) -> bool:
    cls = type(task)
    claimed = getattr(cls, "claims_tool_use", False) or getattr(
        cls, "claims_grounding", False
    )
    if claimed:
        return True
    return metric_claims_tool_use(tuple(trace.metrics))


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
