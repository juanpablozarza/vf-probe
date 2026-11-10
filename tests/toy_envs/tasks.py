"""Two sponsor-citation graders.

Both claim to check the answer and the tool output. ``CleanTask`` does both.
``BlindGroundingTask`` pays the grounding reward for any non-empty reply.
"""

from __future__ import annotations

import json
from typing import ClassVar

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from vf_probe.mutators import declares_evidence

WRONG_ANSWER = "Pfizer"
WRONG_CITATION = "NDA99999"


class SponsorData(vf.TaskData):
    gold_answer: str
    citation_id: str


EXAMPLE = SponsorData.model_validate(
    {
        "id": "sponsor-acme",
        "name": "sponsor-acme",
        "prompt": "Who sponsors the drug in the tool output?",
        "gold_answer": "Acme",
        "citation_id": "NDA12345",
    }
)


class SponsorTask(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    claims_tool_use: ClassVar[bool] = True
    claims_grounding: ClassVar[bool] = True

    @declares_evidence("answer")
    @vf.reward(weight=1.0)
    async def answer(self, trace: vf.Trace) -> float:
        got, _citation = reply_fields(trace.last_reply)
        return float(got == self.data.gold_answer)

    @vf.metric
    async def reply_chars(self, trace: vf.Trace) -> float:
        return float(len(trace.last_reply))


class CleanTask(SponsorTask):
    """Answer matches the gold. Grounding requires the cited id in a tool output."""

    @declares_evidence("observation")
    @vf.reward(weight=1.0)
    async def grounded(self, trace: vf.Trace) -> float:
        _answer, citation = reply_fields(trace.last_reply)
        if citation is None:
            return 0.0
        return float(citation in tool_tokens(trace))


class BlindGroundingTask(SponsorTask):
    """Answer matches the gold. Grounding ignores tool outputs."""

    @declares_evidence("observation")
    @vf.reward(weight=1.0)
    async def grounded(self, trace: vf.Trace) -> float:
        return float(bool(trace.last_reply))


def reply_fields(reply: str) -> tuple[str | None, str | None]:
    """Read ``Answer:`` and ``Citation:`` lines. Runs of whitespace collapse."""
    answer: str | None = None
    citation: str | None = None
    for raw_line in reply.splitlines():
        line = " ".join(raw_line.split())
        if line.startswith("Answer:"):
            answer = line.removeprefix("Answer:").strip() or None
        elif line.startswith("Citation:"):
            citation = line.removeprefix("Citation:").strip() or None
    return answer, citation


def tool_tokens(trace: vf.Trace) -> set[str]:
    tokens: set[str] = set()
    for message in trace.tool_messages:
        tokens.update(content_text(message.content).split())
    return tokens


def lookup_sponsor(data: SponsorData) -> str:
    """Tool result for one sponsor lookup. The probe does not call this."""
    return f"id {data.citation_id} sponsor {data.gold_answer}"


def tool_output(data: SponsorData) -> str:
    return lookup_sponsor(data)


def swapped_tool_output(data: SponsorData) -> str:
    return f"id {WRONG_CITATION} sponsor {data.gold_answer}"


def good_reply(data: SponsorData) -> str:
    return f"Answer: {data.gold_answer}\nCitation: {data.citation_id}"


def spaced_reply(data: SponsorData) -> str:
    return f"Answer:   {data.gold_answer}  \n\n  Citation:   {data.citation_id}  \n"


def wrong_answer_reply(data: SponsorData) -> str:
    return f"Answer: {WRONG_ANSWER}\nCitation: {data.citation_id}"


def toy_trace(task: SponsorTask, reply: str, observation: str) -> vf.Trace:
    """A lookup call, its tool result, then the sampled reply."""
    call_id = "lookup-1"
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig.model_validate({})),
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(
                    tool_calls=[
                        vf.ToolCall(
                            id=call_id,
                            name="lookup_sponsor",
                            arguments=json.dumps({"drug": task.data.id}),
                        )
                    ],
                ),
                sampled=True,
            ),
            vf.MessageNode(
                message=vf.ToolMessage(
                    tool_call_id=call_id,
                    name="lookup_sponsor",
                    content=observation,
                )
            ),
            vf.MessageNode(
                message=vf.AssistantMessage(content=reply),
                sampled=True,
            ),
        ],
    )
