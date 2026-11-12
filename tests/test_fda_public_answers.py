"""Answer edits against FDA-Gym public tasks. Skipped unless the checkout is present.

Hard-set files are not read. The public sheet is FG-9101 through FG-9193.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from vf_probe.answer_edits import answer_replacements, gold_kind

ROOT = Path(os.environ.get("FDA_GYM_DIR", "/tmp/fda-gym"))
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
TRACES = ROOT / "results" / "public-haiku-luna-2026-11-05.jsonl"

pytestmark = pytest.mark.skipif(
    not PUBLIC.is_file() or not TRACES.is_file(),
    reason="FDA-Gym public tasks are not checked out",
)


def test_public_answer_edits_move_only_the_answer_terms() -> None:
    pytest.importorskip("rapidfuzz")
    sys.path.insert(0, str(ROOT))
    from fdagym.prime.episode import grade_steps  # pyright: ignore[reportMissingImports]

    tasks = _tasks()
    other = [
        str(row["gold"]["value"])
        for row in tasks.values()
        if row["gold"]["kind"] == "exact"
    ]
    counts = {"exact": 0, "boolean": 0}
    for trace in _correct_traces(tasks):
        task = tasks[trace["task_id"]]
        kind = task["gold"]["kind"]
        if kind not in counts:
            continue
        answer = trace["final_answer"]["answer"]
        edits = answer_replacements(
            answer,
            gold_kind=gold_kind(_Gold(task)),
            other_golds=other,
        )
        names = [name for name, _replacement in edits]
        if kind == "exact":
            assert "entity_swap" in names
            assert "cross_task_swap" in names
        else:
            assert "yesno_flip" in names
        for _name, replacement in edits:
            mutated = dict(trace["final_answer"])
            mutated["answer"] = replacement
            scored = grade_steps(task, trace["steps"], mutated)
            assert scored["correctness"] == 0.0
            assert scored["format"] == 1.0
            assert scored["citation"] == trace["reward"]["citation"]
            assert scored["efficiency"] == 0.0
        counts[kind] += 1
    assert counts["exact"] > 0
    assert counts["boolean"] == 1


def _tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    with PUBLIC.open() as handle:
        for line in handle:
            row = json.loads(line)
            tasks[row["id"]] = row
    return tasks


def _correct_traces(tasks: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    with TRACES.open() as handle:
        for line in handle:
            row = json.loads(line)
            reward = row["reward"]
            if reward["correctness"] != 1.0 or reward["format"] != 1.0:
                continue
            if tasks[row["task_id"]]["gold"]["kind"] not in {"exact", "boolean"}:
                continue
            rows.append(row)
    return rows


class _Gold:
    def __init__(self, task: dict) -> None:
        self.sheet_row = task

    def model_dump(self) -> dict:
        return {"sheet_row": self.sheet_row}
