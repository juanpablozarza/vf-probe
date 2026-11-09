from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, get_type_hints

import vf_probe
from vf_probe import Mutator


def test_import_and_mutator_shape() -> None:
    assert vf_probe.__name__ == "vf_probe"
    assert is_dataclass(Mutator)
    assert {f.name for f in fields(Mutator)} == {
        "name",
        "target_component",
        "expect",
    }
    hints = get_type_hints(Mutator)
    assert hints["name"] is str
    assert hints["target_component"] is str
    assert set(get_args(hints["expect"])) == {"drop", "hold"}
    assert get_origin(hints["expect"]) is not None

    sig = inspect.signature(Mutator.apply)
    assert list(sig.parameters) == ["self", "trace", "task", "rng"]
    assert sig.return_annotation is not inspect.Signature.empty
