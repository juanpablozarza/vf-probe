"""Load a taskset and record how it scores, without running the probe."""

from __future__ import annotations

from dataclasses import dataclass

from vf_probe.types import AnswerLocator


@dataclass(frozen=True)
class EnvProfile:
    """What the loader can tell from a taskset before scoring.

    The v1 API has no parser. `has_parser` is almost always False; the
    answer locator then falls back to an extract function or the whole
    final reply. Tool-use mutators run only when `claims_grounding` or
    `claims_tool_use` is True. `needs_judge` (~13% of Hub envs) and
    `needs_sandbox` (~25%) are skipped in v1.
    """

    env_id: str
    answer_field: str | None
    has_parser: bool
    weighted_components: tuple[str, ...]
    claims_grounding: bool
    claims_tool_use: bool
    needs_judge: bool
    needs_sandbox: bool


def load(env_id: str) -> EnvProfile:
    """Load tasks and detect answer field, rewards, and skip/claim flags."""
    raise NotImplementedError


_ANSWER_KEYS = frozenset(
    {
        "answer",
        "expected",
        "expected_answer",
        "gold",
        "gold_answer",
        "label",
        "target",
        "value",
    }
)


def locate_answer(trace: object, task: object) -> tuple[str, AnswerLocator]:
    """Return `(answer_text, method)`.

    Fall back parser, then a gold string stored on the task, then the whole
    final reply. The gold string is the longest dumped field whose key is an
    answer name and whose text occurs in the reply. ``citation_id`` and the
    prompt are not answer names.
    """
    reply = _reply_text(trace)
    parsed = _parsed_answer(task, reply)
    if parsed is not None:
        return parsed, "parser"
    for candidate in _answer_strings(task):
        if candidate in reply:
            return candidate, "extract"
    return reply, "final_reply"


def _reply_text(trace: object) -> str:
    reply = getattr(trace, "last_reply", "")
    return reply if isinstance(reply, str) else ""


def _parsed_answer(task: object, reply: str) -> str | None:
    parser = getattr(task, "parser", None)
    parse = getattr(parser, "parse", None)
    if not callable(parse):
        return None
    parsed = parse(reply)
    if not isinstance(parsed, str):
        return None
    text = parsed.strip()
    return text or None


def _answer_strings(task: object) -> list[str]:
    data = getattr(task, "data", None)
    dump_model = getattr(data, "model_dump", None)
    dump = dump_model() if callable(dump_model) else data
    found: list[str] = []

    def walk(node: object, key: str | None) -> None:
        if isinstance(node, str):
            text = node.strip()
            if key in _ANSWER_KEYS and len(text) >= 2:
                found.append(text)
            return
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, child_key if isinstance(child_key, str) else None)
            return
        if isinstance(node, list):
            for child in node:
                walk(child, key)

    walk(dump, None)
    unique = list(dict.fromkeys(found))
    unique.sort(key=len, reverse=True)
    return unique
