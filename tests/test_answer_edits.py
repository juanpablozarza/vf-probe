"""Number, yes/no, entity, and cross-task edits stay inside the answer."""

from __future__ import annotations

import verifiers.v1 as vf

from toy_envs.tasks import EXAMPLE, CleanTask, good_reply, lookup_sponsor, toy_trace
from vf_probe.answer_edits import answer_replacements
from vf_probe.mutators import swap_answer


def test_number_yesno_and_entity_edits() -> None:
    number = answer_replacements("10", gold_kind="number")
    assert number[0][0] == "number_perturb"
    assert number[0][1] != "10"
    assert "10" not in number[0][1]

    flipped = answer_replacements("Yes, bleeding", gold_kind="boolean")
    assert flipped[0] == ("yesno_flip", "no, bleeding")

    entity = answer_replacements("Acme", gold_kind="exact", other_golds=["Pfizer"])
    assert entity[0][0] == "entity_swap"
    assert "Acme" not in entity[0][1]
    assert entity[1] == ("cross_task_swap", "Pfizer")


def test_swap_answer_updates_the_reply_and_the_final_answer() -> None:
    task = CleanTask(EXAMPLE)
    trace = toy_trace(task, good_reply(EXAMPLE), lookup_sponsor(EXAMPLE))
    trace.state = AnswerState(
        final_answer={"answer": EXAMPLE.gold_answer, "citations": [{"id": "NDA12345"}]}
    )
    mutated = swap_answer(trace, task, other_golds=["Pfizer"])
    assert mutated is not None
    assert EXAMPLE.gold_answer not in mutated.last_reply
    assert EXAMPLE.citation_id in mutated.last_reply
    assert _stored_answer(mutated) != EXAMPLE.gold_answer
    assert _stored_citations(mutated) == [{"id": "NDA12345"}]
    assert _stored_answer(trace) == EXAMPLE.gold_answer


class AnswerState(vf.State):
    final_answer: dict | None = None


def _stored_answer(trace: vf.Trace) -> str:
    final = getattr(trace.state, "final_answer", None)
    assert isinstance(final, dict)
    answer = final["answer"]
    assert isinstance(answer, str)
    return answer


def _stored_citations(trace: vf.Trace) -> list[object]:
    final = getattr(trace.state, "final_answer", None)
    assert isinstance(final, dict)
    citations = final["citations"]
    assert isinstance(citations, list)
    return citations
