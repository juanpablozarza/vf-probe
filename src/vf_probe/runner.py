"""Install one env in a throwaway uv venv and run the probe."""

from __future__ import annotations

from pathlib import Path

from vf_probe.types import Report


def run(env_id: str, out: Path | None = None) -> Report:
    """Install `env_id` in a throwaway uv venv with that env's own pins.

    Records the verifiers version on each row. Envs that need a judge or a
    sandbox/external service are marked skipped in v1.
    """
    raise NotImplementedError
