"""Tool-call edits against FDA-Gym public traces. Skipped unless the checkout exists."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf

from vf_probe.mutators import delete_tool_call, perturb_argument, rename_tool

ROOT = Path(os.environ.get("FDA_GYM_DIR", "/tmp/fda-gym"))
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
TRACES = ROOT / "results" / "public-haiku-luna-2026-11-05.jsonl"

pytestmark = pytest.mark.skipif(
    not PUBLIC.is_file() or not TRACES.is_file(),
    reason="FDA-Gym public tasks are not checked out",
)


def test_rename_and_delete_move_fda_citation() -> None:
    pytest.importorskip("rapidfuzz")
    sys.path.insert(0, str(ROOT))
    from fdagym.prime.episode import grade_steps  # pyright: ignore[reportMissingImports]

    tasks = {}
    with PUBLIC.open() as handle:
        for line in handle:
            row = json.loads(line)
            tasks[row["id"]] = row
    checked = 0
    with TRACES.open() as handle:
        for line in handle:
            row = json.loads(line)
            reward = row["reward"]
            if reward["correctness"] != 1.0 or reward["citation"] != 1.0:
                continue
            task = tasks[row["task_id"]]
            trace = _trace(row)
            renamed = rename_tool(trace)
            removed = delete_tool_call(trace)
            changed = perturb_argument(trace)
            assert renamed is not None and removed is not None and changed is not None
            for edited in (renamed, removed, changed):
                steps = getattr(edited.state, "steps", None)
                final_answer = getattr(edited.state, "final_answer", None)
                assert isinstance(steps, list)
                scored = grade_steps(task, steps, final_answer)
                assert scored["citation"] < 1.0 or scored["correctness"] < 1.0
            checked += 1
            if checked == 5:
                break
    assert checked == 5


class _Steps(vf.State):
    steps: list | None = None
    final_answer: dict | None = None


def _trace(row: dict) -> vf.Trace:
    calls = []
    messages = []
    for index, step in enumerate(row["steps"]):
        tool = step.get("tool")
        if not isinstance(tool, str) or tool == "submit_answer":
            continue
        call_id = f"c{index}"
        raw_arguments = step.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        calls.append(
            vf.ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))
        )
        messages.append(
            vf.MessageNode(
                message=vf.ToolMessage(
                    tool_call_id=call_id,
                    name=tool,
                    content=json.dumps(step.get("output")),
                )
            )
        )
    return vf.Trace(
        task=vf.TraceTask(type="FDA", data=vf.TaskData.model_validate({})),
        agent=vf.AgentInfo(config=vf.AgentConfig.model_validate({})),
        state=_Steps(steps=row["steps"], final_answer=row["final_answer"]),
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(tool_calls=calls),
                sampled=True,
            ),
            *messages,
        ],
    )
