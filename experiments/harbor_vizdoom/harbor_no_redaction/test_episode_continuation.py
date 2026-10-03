"""Offline checks for the Harbor Codex continuation shim."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from episode_continuation import _resume_command, continue_codex_episode


class _FakeEnvironment:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = iter(statuses)
        self.reads = 0

    async def service_exec(self, command: str, *, service: str) -> SimpleNamespace:
        assert command == "cat /vizdoom-results/result.json"
        assert service == "vizdoom"
        self.reads += 1
        return SimpleNamespace(
            return_code=0,
            stdout=json.dumps({"status": next(self.statuses)}),
        )


class ContinuationTests(unittest.TestCase):
    command = (
        "codex exec --json --model gpt-5.6-terra -- 'Start' "
        "2>&1 </dev/null | tee /installed-agent/codex.txt"
    )

    def test_resume_command_preserves_session_and_appends_log(self) -> None:
        resumed = _resume_command(self.command)
        self.assertIn("codex exec resume --last ", resumed)
        self.assertIn("--model gpt-5.6-terra", resumed)
        self.assertIn("tee -a", resumed)
        self.assertNotIn("-- 'Start'", resumed)

    def test_early_final_resumes_same_episode(self) -> None:
        calls: list[str] = []

        async def original(_agent, _environment, *, command, **_kwargs):
            calls.append(command)
            return len(calls)

        environment = _FakeEnvironment(["running", "running", "timeout"])
        agent = SimpleNamespace(logger=SimpleNamespace(info=lambda *_a: None))
        result = asyncio.run(
            continue_codex_episode(original, agent, environment, self.command)
        )
        self.assertEqual(result, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(environment.reads, 3)
        self.assertTrue(all("resume --last" in call for call in calls[1:]))

    def test_completed_episode_does_not_resume(self) -> None:
        calls: list[str] = []

        async def original(_agent, _environment, *, command, **_kwargs):
            calls.append(command)

        environment = _FakeEnvironment(["success"])
        agent = SimpleNamespace(logger=SimpleNamespace(info=lambda *_a: None))
        asyncio.run(continue_codex_episode(original, agent, environment, self.command))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
