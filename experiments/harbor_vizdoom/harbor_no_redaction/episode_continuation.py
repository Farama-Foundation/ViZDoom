"""Keep a Harbor Codex turn attached to ViZDoom until the episode ends.

Harbor 0.22 runs one ``codex exec`` turn per trial. A model can answer in
natural language while ViZDoom still reports ``status: running``. Resume the
same Codex session, with the same live MCP sidecar, instead of scoring that as
an episode outcome. No action is taken by the harness itself.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Awaitable, Callable

from harbor.models.trial.paths import EnvironmentPaths

CONTINUATION_PROMPT = (
    "Your previous response ended the Codex turn, but the ViZDoom episode is "
    "still running. Resume the same episode now: call get_current_state, "
    "interpret the image and state, then use make_action. Keep playing until "
    "the tool returns done=true. Do not give a final response while done=false, "
    "even if the objective seems difficult or unlikely."
)
MAX_CONTINUATIONS = 20


def _is_game_command(command: str) -> bool:
    return "codex exec " in command and " | tee " in command


def _resume_command(command: str) -> str:
    prefix, _separator, suffix = command.partition(" -- ")
    if not prefix or " | tee " not in suffix:
        raise RuntimeError("Unexpected Harbor Codex command format")
    tee_path = (EnvironmentPaths.agent_dir / "codex.txt").as_posix()
    return (
        prefix.replace("codex exec ", "codex exec resume --last ", 1)
        + " -- "
        + shlex.quote(CONTINUATION_PROMPT)
        + " 2>&1 </dev/null | tee -a "
        + shlex.quote(tee_path)
    )


async def _episode_status(environment: Any) -> str:
    result = await environment.service_exec(
        "cat /vizdoom-results/result.json", service="vizdoom"
    )
    if result.return_code != 0:
        raise RuntimeError("Could not inspect live ViZDoom episode status")
    status = json.loads(result.stdout)["status"]
    if not isinstance(status, str):
        raise RuntimeError("ViZDoom returned a non-string episode status")
    return status


async def continue_codex_episode(
    original_exec: Callable[..., Awaitable[Any]],
    agent: Any,
    environment: Any,
    command: str,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_sec: int | None = None,
) -> Any:
    kwargs = {"env": env, "cwd": cwd, "timeout_sec": timeout_sec}
    result = await original_exec(agent, environment, command=command, **kwargs)
    if not _is_game_command(command):
        return result

    for attempt in range(MAX_CONTINUATIONS + 1):
        status = await _episode_status(environment)
        if status != "running":
            return result
        if attempt == MAX_CONTINUATIONS:
            raise RuntimeError(
                f"Codex ended {MAX_CONTINUATIONS + 1} turns while ViZDoom "
                "was still running; trial must not be scored as an ending"
            )
        agent.logger.info(
            "Codex turn ended before ViZDoom; resuming same episode (%d/%d)",
            attempt + 1,
            MAX_CONTINUATIONS,
        )
        result = await original_exec(
            agent, environment, command=_resume_command(command), **kwargs
        )
    return result
