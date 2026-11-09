"""Score the two toy graders through ``Task.score``."""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf

from toy_envs.tasks import (
    EXAMPLE,
    WRONG_ANSWER,
    WRONG_CITATION,
    BlindGroundingTask,
    CleanTask,
    SponsorTask,
    good_reply,
    spaced_reply,
    swapped_tool_output,
    tool_output,
    toy_trace,
    wrong_answer_reply,
)


def test_both_tasks_claim_tool_use() -> None:
    assert CleanTask.claims_tool_use is True
    assert CleanTask.claims_grounding is True
    assert BlindGroundingTask.claims_tool_use is True
    assert BlindGroundingTask.claims_grounding is True
    assert EXAMPLE.gold_answer != WRONG_ANSWER
    assert EXAMPLE.citation_id != WRONG_CITATION


def test_clean_grades_the_evidence_each_reward_claims() -> None:
    base, changed_answer, swapped_obs, spaced = asyncio.run(_clean_cases())

    assert _scores(base) == {"answer": 1.0, "grounded": 1.0}
    assert base.reward == 2.0
    assert _weights(base) == {"answer": 1.0, "grounded": 1.0}
    assert base.metrics["reply_chars"] == float(len(base.last_reply))
    assert "reply_chars" not in base.rewards

    assert _scores(changed_answer) == {"answer": 0.0, "grounded": 1.0}
    assert changed_answer.reward == 1.0

    assert _scores(swapped_obs) == {"answer": 1.0, "grounded": 0.0}
    assert swapped_obs.reward == 1.0

    assert _scores(spaced) == {"answer": 1.0, "grounded": 1.0}
    assert spaced.metrics["reply_chars"] != base.metrics["reply_chars"]


def test_blind_grounding_ignores_the_tool_output() -> None:
    base, changed_answer, swapped_obs = asyncio.run(_blind_cases())

    assert _scores(base) == {"answer": 1.0, "grounded": 1.0}
    assert _scores(changed_answer) == {"answer": 0.0, "grounded": 1.0}
    assert _scores(swapped_obs) == {"answer": 1.0, "grounded": 1.0}
    assert swapped_obs.reward == 2.0
    assert "reply_chars" not in swapped_obs.rewards


async def _clean_cases() -> tuple[vf.Trace, vf.Trace, vf.Trace, vf.Trace]:
    task = CleanTask(EXAMPLE)
    observation = tool_output(EXAMPLE)
    return (
        await _score(task, good_reply(EXAMPLE), observation),
        await _score(task, wrong_answer_reply(EXAMPLE), observation),
        await _score(task, good_reply(EXAMPLE), swapped_tool_output(EXAMPLE)),
        await _score(task, spaced_reply(EXAMPLE), observation),
    )


async def _blind_cases() -> tuple[vf.Trace, vf.Trace, vf.Trace]:
    task = BlindGroundingTask(EXAMPLE)
    observation = tool_output(EXAMPLE)
    return (
        await _score(task, good_reply(EXAMPLE), observation),
        await _score(task, wrong_answer_reply(EXAMPLE), observation),
        await _score(task, good_reply(EXAMPLE), swapped_tool_output(EXAMPLE)),
    )


async def _score(task: SponsorTask, reply: str, observation: str) -> vf.Trace:
    trace = toy_trace(task, reply, observation)
    await task.score(trace, runtime=None)
    first = _scores(trace)
    first_metrics = dict(trace.metrics)
    await task.score(trace, runtime=None)
    assert _scores(trace) == first
    assert trace.metrics == first_metrics
    return trace


def _scores(trace: vf.Trace) -> dict[str, float]:
    return {name: _score_of(trace, name) for name in ("answer", "grounded")}


def _weights(trace: vf.Trace) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name in ("answer", "grounded"):
        reward = trace.rewards[name]
        assert reward is not None
        weights[name] = reward.weight
    return weights


def _score_of(trace: vf.Trace, name: str) -> float:
    reward = trace.rewards[name]
    assert reward is not None
    return reward.score
