"""Disable Harbor's destructive post-trial result redaction.

Python imports ``sitecustomize`` automatically during interpreter startup when
its directory is on ``PYTHONPATH``. The ViZDoom launcher injects this directory
only into the Harbor subprocess, so the installed Harbor package is unchanged.

This intentionally disables secret scrubbing for every textual file in a
trial directory. Do not publish a resulting job directory without inspecting
it for credentials first.
"""

from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.trial.trial import Trial

from episode_continuation import continue_codex_episode


def _skip_jobs_dir_scrubbing(_trial: Any) -> None:
    """Keep Harbor trial files byte-for-byte as originally written."""


setattr(Trial, "_scrub_jobs_dir", _skip_jobs_dir_scrubbing)


_original_codex_exec_as_agent = Codex.exec_as_agent


async def _exec_codex_until_episode_ends(
    self: Codex,
    environment: Any,
    command: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_sec: int | None = None,
) -> Any:
    return await continue_codex_episode(
        _original_codex_exec_as_agent,
        self,
        environment,
        command,
        env=env,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )


setattr(Codex, "exec_as_agent", _exec_codex_until_episode_ends)
