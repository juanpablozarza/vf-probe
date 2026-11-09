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
    """Deep-copy `trace`, await Task.score, repeat `k` times."""
    raise NotImplementedError
