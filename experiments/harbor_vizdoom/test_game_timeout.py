"""Check task configuration and the real Doom engine's simulated-time cutoff."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_tasks import (
    COMPOSE_TEMPLATE,
    INSTRUCTIONS_DIR,
    _render,
    _template_values,
    load_task_definitions,
)
from random_agent_harness import _run_task


class DoomTimeoutTests(unittest.TestCase):
    def test_default_scenario_timeout_is_preserved(self) -> None:
        task = next(task for task in load_task_definitions() if task["slug"] == "basic")
        self.assertNotIn("episode_timeout_seconds", task)
        with tempfile.TemporaryDirectory() as temporary:
            outcome = _run_task(
                task,
                Path(temporary),
                seed=12345,
                max_actions=80,
                button_probability=0,
                memory_every=0,
            )
            task_dir = Path(temporary) / task["slug"]
            diagnostics = "\n".join(
                path.read_text(encoding="utf-8")
                for path in task_dir.glob("*.stderr.log")
            )
            self.assertTrue(outcome["passed"], diagnostics)
            result = json.loads(
                (task_dir / "vizdoom-results/result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["episode_timeout_seconds"], 0)
            self.assertEqual(result["elapsed_game_tics"], 300)
            self.assertEqual(result["decisions"], 75)

    def test_e1m1_has_ten_minute_budget(self) -> None:
        task = next(
            task for task in load_task_definitions() if task["slug"] == "doom-e1m1"
        )
        self.assertEqual(task["episode_timeout_seconds"], 600)
        self.assertGreaterEqual(task["max_decisions"] * 4, 600 * 35)
        values = _template_values(task)
        compose = _render(COMPOSE_TEMPLATE, values)
        instruction = _render(INSTRUCTIONS_DIR / "doom.md.template", values)
        self.assertIn('VIZDOOM_EPISODE_TIMEOUT_SECONDS: "600"', compose)
        self.assertIn("600-second in-game", instruction)

    def test_all_requested_maps_cover_their_game_time_limits(self) -> None:
        limits = {
            "doom-e1m1": 600,
            "doom-e1m2": 1200,
            "doom2-map01": 600,
            "doom2-map02": 1200,
        }
        for task in load_task_definitions():
            if task["slug"] not in limits:
                continue
            with self.subTest(task=task["slug"]):
                seconds = limits[task["slug"]]
                self.assertEqual(task["episode_timeout_seconds"], seconds)
                self.assertEqual(task["max_decisions"] * 4, seconds * 35)
                self.assertTrue(task["require_automap"])
                self.assertEqual(task["instruction_template"], "doom.md.template")
                values = _template_values(task)
                self.assertIn(
                    f'VIZDOOM_EPISODE_TIMEOUT_SECONDS: "{seconds}"',
                    _render(COMPOSE_TEMPLATE, values),
                )

    def test_engine_timeout_finalizes_video_and_passes_verifier(self) -> None:
        task = next(
            task for task in load_task_definitions() if task["slug"] == "doom-e1m1"
        )
        task = {**task, "episode_timeout_seconds": 1, "max_decisions": 9}
        with tempfile.TemporaryDirectory() as temporary:
            outcome = _run_task(
                task,
                Path(temporary),
                seed=12345,
                max_actions=20,
                button_probability=0,
                memory_every=0,
            )
            task_dir = Path(temporary) / task["slug"]
            diagnostics = "\n".join(
                path.read_text(encoding="utf-8")
                for path in task_dir.glob("*.stderr.log")
            )
            self.assertTrue(outcome["passed"], diagnostics)
            result = json.loads(
                (task_dir / "vizdoom-results/result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["elapsed_game_tics"], 35)
            self.assertTrue(result["elapsed_game_tics_exact"])
            self.assertEqual(result["decisions"], 9)
            self.assertTrue(result["video_finalized"])


if __name__ == "__main__":
    unittest.main()
