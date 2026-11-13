"""Tool-call edits that a call checker and a result checker can both feel."""

from __future__ import annotations

import asyncio
import json

import verifiers.v1 as vf

from toy_envs.tasks import (
    EXAMPLE,
    CleanTask,
    SponsorData,
    good_reply,
    lookup_sponsor,
    toy_trace,
)
from vf_probe.mutators import (
    delete_tool_call,
    perturb_argument,
    rename_tool,
    swap_arguments,
)
from vf_probe.probe import probe


def test_call_edits_change_the_call_and_leave_the_result() -> None:
    trace = _weather_trace()
    renamed = rename_tool(trace)
    changed = perturb_argument(trace)
    swapped = swap_arguments(trace)
    removed = delete_tool_call(trace)
    assert renamed is not None and changed is not None
    assert swapped is not None and removed is not None
    renamed_call = _call(renamed)
    changed_call = _call(changed)
    swapped_call = _call(swapped)
    assert renamed_call is not None and changed_call is not None
    assert swapped_call is not None
    assert renamed_call.name == "get_weather_other"
    assert json.loads(renamed_call.arguments)["city"] == "Boston"
    assert json.loads(changed_call.arguments)["city"] != "Boston"
    assert changed_call.name == "get_weather"
    assert _result(changed) == _result(trace)
    args = json.loads(swapped_call.arguments)
    assert args["city"] == "Mon"
    assert args["day"] == "Boston"
    assert _call(removed) is None
    assert removed.tool_messages == []


def test_probe_accepts_a_checker_that_reads_the_call() -> None:
    task = CallChecker(EXAMPLE)
    result = asyncio.run(probe(task, _weather_trace()))
    assert result.findings == ()
    assert result.reason is None
    assert result.clean is True


class CallChecker(vf.Task[SponsorData, vf.State, vf.TaskConfig]):
    """Pays 1 only when the call is get_weather(city=Boston, day=Mon)."""

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        call = _call(trace)
        if call is None or call.name != "get_weather":
            return 0.0
        args = json.loads(call.arguments)
        return float(args.get("city") == "Boston" and args.get("day") == "Mon")


def _weather_trace() -> vf.Trace:
    base = toy_trace(CleanTask(EXAMPLE), good_reply(EXAMPLE), lookup_sponsor(EXAMPLE))
    call = vf.ToolCall(
        id="weather-1",
        name="get_weather",
        arguments=json.dumps({"city": "Boston", "day": "Mon"}),
    )
    base.nodes = [
        vf.MessageNode(
            message=vf.AssistantMessage(tool_calls=[call]),
            sampled=True,
        ),
        vf.MessageNode(
            message=vf.ToolMessage(
                tool_call_id="weather-1",
                name="get_weather",
                content="sunny",
            )
        ),
    ]
    return base


def _call(trace: vf.Trace) -> vf.ToolCall | None:
    for node in trace.nodes:
        message = node.message
        if isinstance(message, vf.AssistantMessage) and message.tool_calls:
            return message.tool_calls[0]
    return None


def _result(trace: vf.Trace) -> str:
    for message in trace.tool_messages:
        if isinstance(message.content, str):
            return message.content
    return ""
