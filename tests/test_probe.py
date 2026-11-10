"""The probe separates the clean toy from the blind one without env-specific text."""

from __future__ import annotations

import asyncio
from typing import Any

import verifiers.v1 as vf

from toy_envs.tasks import (
    EXAMPLE,
    BlindGroundingTask,
    CleanTask,
    SponsorData,
    SponsorTask,
    good_reply,
    lookup_sponsor,
    toy_trace,
)
from vf_probe.loader import locate_answer
from vf_probe.mutators import (
    OBSERVATION_REDACTION,
    PlannedEdit,
    declares_evidence,
    select_mutators,
    swap_answer,
    swap_observation,
)
from vf_probe.probe import probe
from vf_probe.support import NEEDS_JUDGE, NEEDS_SANDBOX, NOT_TEXT_SCORABLE
from vf_probe.types import ProbeResult


def test_locate_answer_reads_a_nested_gold_value() -> None:
    class Data:
        def model_dump(self) -> dict[str, object]:
            return {
                "prompt": "Who sponsors it?",
                "citation_id": EXAMPLE.citation_id,
                "gold": {"kind": "exact", "value": EXAMPLE.gold_answer},
            }

    class Task:
        data = Data()

    class Trace:
        last_reply = good_reply(EXAMPLE)

    assert locate_answer(Trace(), Task()) == (EXAMPLE.gold_answer, "extract")


def test_mutators_follow_evidence_instead_of_reward_names() -> None:
    declared = select_mutators(CleanTask(EXAMPLE), ("answer", "grounded"))
    assert _edits(declared, attributed=True) == [
        ("swap_answer", "answer"),
        ("swap_observation", "grounded"),
    ]

    read = select_mutators(ReadsState(EXAMPLE), ("score_total", "format_only"))
    assert _edits(read, attributed=True) == [
        ("swap_answer", "score_total"),
        ("swap_observation", "score_total"),
    ]
    assert _edits(read, attributed=False) == [
        ("swap_answer", "format_only"),
        ("swap_observation", "format_only"),
    ]

    misnamed = select_mutators(Misnamed(EXAMPLE), ("answer", "other"))
    assert _edits(misnamed, attributed=True) == [("swap_observation", "answer")]

    single = select_mutators(Constant(EXAMPLE), ("score",))
    assert _edits(single, attributed=True) == [
        ("swap_answer", "score"),
        ("swap_observation", "score"),
    ]
    declared = select_mutators(Declared(EXAMPLE), ("grounded",))
    assert _edits(declared, attributed=True) == [("swap_answer", "grounded")]


def _edits(plans: list[PlannedEdit], *, attributed: bool) -> list[tuple[str, str]]:
    return [
        (plan.mutator.name, plan.mutator.target_component)
        for plan in plans
        if plan.attributed is attributed
    ]


def test_mutations_keep_the_evidence_they_do_not_target() -> None:
    task = CleanTask(EXAMPLE)
    trace = toy_trace(task, good_reply(EXAMPLE), lookup_sponsor(EXAMPLE))
    changed_answer = swap_answer(trace, task)
    changed_obs = swap_observation(trace, task)
    assert changed_answer is not None
    assert changed_obs is not None
    assert EXAMPLE.gold_answer not in changed_answer.last_reply
    assert EXAMPLE.citation_id in changed_answer.last_reply
    assert lookup_sponsor(EXAMPLE) in _tool_text(changed_answer)
    assert changed_obs.last_reply == trace.last_reply
    assert _tool_text(changed_obs) == OBSERVATION_REDACTION
    assert EXAMPLE.citation_id not in _tool_text(changed_obs)


def test_probe_flags_only_the_blind_grounding_reward() -> None:
    clean, blind, answer_only = asyncio.run(_probe_all())
    assert clean.clean is True
    assert clean.reason is None
    assert clean.findings == ()
    assert len(blind.findings) == 1
    finding = blind.findings[0]
    assert blind.clean is False
    assert blind.reason is None
    assert finding.mutator == "swap_observation"
    assert finding.component == "grounded"
    assert finding.base == 1.0
    assert finding.mutant == 1.0
    assert answer_only.clean is True
    assert answer_only.findings == ()


async def _probe_all() -> tuple[ProbeResult, ProbeResult, ProbeResult]:
    observation = lookup_sponsor(EXAMPLE)
    reply = good_reply(EXAMPLE)
    clean = CleanTask(EXAMPLE)
    blind = BlindGroundingTask(EXAMPLE)
    answer_only = AnswerOnly(EXAMPLE)
    return (
        await probe(clean, toy_trace(clean, reply, observation)),
        await probe(blind, toy_trace(blind, reply, observation)),
        await probe(answer_only, toy_trace(answer_only, reply, observation)),
    )


class AnswerOnly(SponsorTask):
    """Pays the answer only, and does not claim to check tool output."""

    claims_tool_use = False
    claims_grounding = False


def test_unsupported_environments_return_a_reason() -> None:
    sandbox, judged, contained, zero, missing, mixed = asyncio.run(_unsupported())
    assert sandbox.clean is False
    assert sandbox.reason == "needs_sandbox"
    assert sandbox.message == NEEDS_SANDBOX
    assert sandbox.skips[0].component == "tests"
    assert SandboxOnly.calls == 0

    assert judged.clean is False
    assert judged.reason == "needs_judge"
    assert judged.message == NEEDS_JUDGE
    assert Judged.calls == 0

    assert contained.reason == "needs_sandbox"
    assert contained.message == NEEDS_SANDBOX
    assert ContainerTask.calls == 0

    assert zero.reason == "not_text_scorable"
    assert zero.message == NOT_TEXT_SCORABLE
    assert zero.clean is False
    assert missing.reason == "not_text_scorable"
    assert missing.message == NOT_TEXT_SCORABLE

    assert mixed.clean is True
    assert mixed.reason is None
    assert mixed.findings == ()
    assert mixed.skips[0].component == "tests"
    assert mixed.skips[0].reason == "needs_sandbox"


async def _unsupported() -> tuple[
    ProbeResult,
    ProbeResult,
    ProbeResult,
    ProbeResult,
    ProbeResult,
    ProbeResult,
]:
    SandboxOnly.calls = 0
    Judged.calls = 0
    ContainerTask.calls = 0
    reply = good_reply(EXAMPLE)
    observation = lookup_sponsor(EXAMPLE)
    trace = toy_trace(CleanTask(EXAMPLE), reply, observation)
    judged = Judged(
        EXAMPLE,
        vf.TaskConfig.model_validate(
            {"judges": [{"id": "decision", "name": "grader"}]}
        ),
    )
    zero = ZeroGrade(EXAMPLE)
    text = TextAndSandbox(EXAMPLE)
    return (
        await probe(SandboxOnly(EXAMPLE), trace),
        await probe(judged, trace),
        await probe(ContainerTask(EXAMPLE), trace),
        await probe(zero, toy_trace(zero, reply, observation)),
        await probe(NoReward(EXAMPLE), trace),
        await probe(text, toy_trace(text, reply, observation)),
    )


class SandboxOnly(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    calls = 0

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        type(self).calls += 1
        await super().score(trace, runtime=runtime)

    @vf.reward(weight=1.0)
    async def tests(self, trace: vf.Trace, runtime: object) -> float:
        return 1.0


class ContainerTask(CleanTask):
    NEEDS_CONTAINER = True
    calls = 0

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        type(self).calls += 1
        await super().score(trace, runtime=runtime)


class Judged(CleanTask):
    calls = 0

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        type(self).calls += 1
        await super().score(trace, runtime=runtime)


class ZeroGrade(SponsorTask):
    claims_tool_use = False
    claims_grounding = False

    @vf.reward(weight=1.0)
    async def answer(self, trace: vf.Trace) -> float:
        return 0.0


class NoReward(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    """Scores nothing. There is no text signal to test."""


class TextAndSandbox(SponsorTask):
    claims_tool_use = False
    claims_grounding = False

    @vf.reward(weight=1.0)
    async def tests(self, trace: vf.Trace, runtime: object) -> float:
        return 1.0


def grade_steps(state: Any) -> float:
    steps = state.steps
    final_answer = state.final_answer
    return float(steps is not None or final_answer is not None)


class ReadsState(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    """score_total reads steps and final_answer through one helper."""

    @vf.reward(weight=1.0)
    async def score_total(self, trace: vf.Trace) -> float:
        return grade_steps(trace.state)

    @vf.reward(weight=1.0)
    async def format_only(self, trace: vf.Trace) -> float:
        return 1.0


class Misnamed(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    """The reward is named answer and only reads tool results."""

    @vf.reward(weight=1.0)
    async def answer(self, trace: vf.Trace) -> float:
        return float(bool(trace.tool_messages))

    @vf.reward(weight=1.0)
    async def other(self, trace: vf.Trace) -> float:
        return 1.0


class Declared(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    """Declared as answer evidence. The body still reads tool results."""

    @declares_evidence("answer")
    @vf.reward(weight=1.0)
    async def grounded(self, trace: vf.Trace) -> float:
        return float(bool(trace.tool_messages))


class Constant(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def score(self, trace: vf.Trace) -> float:
        return 1.0


class Opaque(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def alpha(self, trace: vf.Trace) -> float:
        return 1.0

    @vf.reward(weight=1.0)
    async def beta(self, trace: vf.Trace) -> float:
        return 1.0


def test_opaque_rewards_keep_their_scores_and_are_not_findings() -> None:
    task = Opaque(EXAMPLE)
    trace = toy_trace(CleanTask(EXAMPLE), good_reply(EXAMPLE), lookup_sponsor(EXAMPLE))
    result = asyncio.run(probe(task, trace))
    assert result.findings == ()
    assert result.clean is False
    assert result.reason is None
    assert {(note.mutator, note.component) for note in result.unattributed} == {
        ("swap_answer", "alpha"),
        ("swap_observation", "alpha"),
        ("swap_answer", "beta"),
        ("swap_observation", "beta"),
    }


def _tool_text(trace: vf.Trace) -> str:
    parts: list[str] = []
    for message in trace.tool_messages:
        content = message.content
        parts.append(content if isinstance(content, str) else "")
    return "\n".join(parts)
