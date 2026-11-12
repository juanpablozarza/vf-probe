"""Answer-span edits. Each one changes the answer and leaves the citation alone."""

from __future__ import annotations

import re
from collections.abc import Sequence

_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_YESNO = re.compile(r"^(yes|no)\b", re.IGNORECASE)
_GOLD_KINDS = frozenset({"exact", "number", "boolean"})


def answer_replacements(
    answer: str,
    *,
    gold_kind: str | None = None,
    other_golds: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Return `(edit_name, replacement)` for the edits that apply to `answer`."""
    text = answer.strip()
    if not text:
        return []
    kind = _kind(text, gold_kind)
    edits: list[tuple[str, str]] = []
    primary = _primary(text, kind)
    if primary is not None and _norm(primary) != _norm(text):
        edits.append((_NAME[kind], primary))
    for other in other_golds:
        replacement = str(other).strip()
        if not replacement or _norm(replacement) == _norm(text):
            continue
        if any(existing == replacement for _, existing in edits):
            continue
        edits.append(("cross_task_swap", replacement))
        break
    return edits


def gold_kind(task: object) -> str | None:
    """`exact`, `number`, or `boolean` when the task stores a gold spec."""
    data = getattr(task, "data", None)
    dump_model = getattr(data, "model_dump", None)
    dump = dump_model() if callable(dump_model) else data
    found = _find_gold(dump)
    if found is None:
        return None
    kind = found.get("kind")
    return kind if isinstance(kind, str) else None


def _primary(text: str, kind: str) -> str | None:
    if kind == "number":
        return _perturb_number(text)
    if kind == "yesno":
        return _flip_yesno(text)
    return _swap_entity(text)


def _kind(text: str, gold_kind_name: str | None) -> str:
    if gold_kind_name == "boolean" or _YESNO.match(text):
        return "yesno"
    if gold_kind_name == "number" or _NUMBER.match(text.replace(" ", "")):
        return "number"
    return "entity"


_NAME = {"number": "number_perturb", "yesno": "yesno_flip", "entity": "entity_swap"}


def _perturb_number(text: str) -> str | None:
    number = _as_number(text)
    if number is None:
        return None
    shifted = number + max(1.0, abs(number))
    rendered = _render_number(shifted, text)
    if text.strip() in rendered:
        rendered = _render_number(shifted + max(1.0, abs(shifted)), text)
    return rendered


def _flip_yesno(text: str) -> str | None:
    match = _YESNO.match(text)
    if match is None:
        return None
    flipped = "no" if match.group(1).lower() == "yes" else "yes"
    return flipped + text[match.end() :]


def _swap_entity(text: str) -> str:
    reversed_text = text[::-1]
    if _norm(reversed_text) != _norm(text) and text not in reversed_text:
        return reversed_text
    if text not in "other-entity":
        return "other-entity"
    return "other-entity-2"


def _as_number(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", ""))
    except ValueError:
        return None


def _render_number(number: float, original: str) -> str:
    if "." not in original and number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .")


def _find_gold(node: object) -> dict[str, object] | None:
    if isinstance(node, dict):
        kind = node.get("kind")
        if isinstance(kind, str) and kind in _GOLD_KINDS and "value" in node:
            return node
        for child in node.values():
            found = _find_gold(child)
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for child in node:
            found = _find_gold(child)
            if found is not None:
                return found
    return None
