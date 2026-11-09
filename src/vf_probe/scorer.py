"""Score a (possibly mutated) trace through ``Task.score``.

Import path (verifiers 0.4.1): ``verifiers.v1.task.Task.score``.
Also exported as ``verifiers.v1.Task``.

Signature::

    async def score(
        self,
        trace: Trace,
        runtime: Runtime | None = None,
        deployment: Deployment | None = None,
    ) -> None

``score`` is async and returns None. It writes onto the trace in place
(deep-copy the trace, not the task):

- ``trace.rewards``: ``dict[str, Reward | None]``, named weighted
  components. ``Reward.score`` is the raw value, ``Reward.weight`` the
  weight, ``Reward.value`` is ``score * weight``.
- ``trace.reward``: sum of ``Reward.value`` (the total score).
- ``trace.metrics``: ``dict[str, float | None]``, unweighted observations.

v1 flags use total-score sensitivity (``trace.reward``). Per-component
attribution from ``trace.rewards`` only when two or more components have
nonzero weight. Zero-weight rewards and ``trace.metrics`` are evidence
only, never flags.

0.4.0's ``score`` has no ``deployment`` argument; pass ``runtime=None``
on both. ``Task.rescore(trace)`` (0.4.1 only) clears rewards/metrics and
re-runs offline signals. Hooks that require a runtime or deployment are
skipped when those are None, which is how v1 detects not-text-scorable
and needs-sandbox envs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.task import Task
    from verifiers.v1.trace import Trace


async def score(
    task: Task,
    trace: Trace,
    *,
    k: int = 1,
    eps: float = 0.0,
    runtime: Runtime | None = None,
) -> Trace:
    """Deep-copy `trace`, await ``Task.score(runtime=None)``, repeat `k` times.

    ``runtime`` is accepted and passed through. Do not pass ``deployment``:
    verifiers 0.4.0 has no such argument. When `k` is greater than 1, a
    component score that moves by more than `eps` raises ``RuntimeError``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    last: Trace | None = None
    first: dict[str, float | None] | None = None
    for _ in range(k):
        copied = trace.model_copy(deep=True)
        await task.score(copied, runtime=runtime)
        seen = _component_scores(copied)
        if first is None:
            first = seen
        elif _moved(first, seen, eps):
            raise RuntimeError("score changed across repeats")
        last = copied
    if last is None:
        raise RuntimeError("score did not run")
    return last


def _component_scores(trace: vf.Trace) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for name, reward in trace.rewards.items():
        scores[name] = None if reward is None else reward.score
    return scores


def _moved(
    first: dict[str, float | None],
    seen: dict[str, float | None],
    eps: float,
) -> bool:
    if first.keys() != seen.keys():
        return True
    for name, before in first.items():
        after = seen[name]
        if before is None or after is None:
            if before is not after:
                return True
            continue
        if abs(before - after) > eps:
            return True
    return False
