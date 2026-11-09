"""The probe separates the clean toy from the blind one without env-specific text."""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf

from toy_envs.tasks import (
    EXAMPLE,
    BlindGroundingTask,
    CleanTask,
    SponsorTask,
    good_reply,
    lookup_sponsor,
    toy_trace,
)
from vf_probe.loader import locate_answer
from vf_probe.mutators import (
    OBSERVATION_REDACTION,
    select_mutators,
    swap_answer,
    swap_observation,
)
from vf_probe.probe import probe
from vf_probe.types import Finding


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


def test_mutators_choose_components_by_name_and_claim() -> None:
    paired = select_mutators(("answer", "grounded"), claims_tool_use=True)
    assert [(spec.name, spec.target_component) for spec in paired] == [
        ("swap_answer", "answer"),
        ("swap_observation", "grounded"),
    ]
    single = select_mutators(("score",), claims_tool_use=True)
    assert [(spec.name, spec.target_component) for spec in single] == [
        ("swap_answer", "score"),
        ("swap_observation", "score"),
    ]
    answer_only = select_mutators(("answer",), claims_tool_use=False)
    assert [(spec.name, spec.target_component) for spec in answer_only] == [
        ("swap_answer", "answer"),
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
    assert clean == []
    assert len(blind) == 1
    finding = blind[0]
    assert finding.mutator == "swap_observation"
    assert finding.component == "grounded"
    assert finding.base == 1.0
    assert finding.mutant == 1.0
    assert answer_only == []


async def _probe_all() -> tuple[list[Finding], list[Finding], list[Finding]]:
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


def _tool_text(trace: vf.Trace) -> str:
    parts: list[str] = []
    for message in trace.tool_messages:
        content = message.content
        parts.append(content if isinstance(content, str) else "")
    return "\n".join(parts)
