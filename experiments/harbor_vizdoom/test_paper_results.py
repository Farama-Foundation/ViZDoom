"""Checks against mixing unmatched or incomplete episodes in the paper table."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from export_paper_results import (
    LEGACY_TASK_SETTINGS,
    add_additional_summaries,
    add_level_summaries,
    combine_summaries,
    render_status,
    render_table,
)
from run_pilot import BASELINE_MODELS, MODELS, TASKS


def summary_fixture(model: str) -> dict:
    return {
        "protocol": {
            "models": [model],
            "seeds": [1, 2, 3, 4, 5],
            "tasks": list(TASKS),
            "reasoning": "medium",
            "auth": "subscription",
            "frame_skip": 4,
            "max_action_repeat": 2,
            "created_at": "test-only",
            "task_settings": copy.deepcopy(LEGACY_TASK_SETTINGS),
        },
        "indexed_trials": 20,
        "expected_trials": 20,
        "incomplete_retry_attempts": 0,
        "conditions": {
            model: {
                task: {
                    "expected": 5,
                    "valid_natural": 5,
                    "unresolved_invalid": [],
                    "per_seed_primary": {str(seed): 1 for seed in range(1, 6)},
                    "seeds": {str(seed): "success" for seed in range(1, 6)},
                    "primary_mean": 1,
                    "successes": 5 if task in ("basic", "my-way-home") else None,
                }
                for task in TASKS
            }
        },
    }


def level_fixture(models: tuple = BASELINE_MODELS) -> dict:
    protocol = summary_fixture(BASELINE_MODELS[0])["protocol"]
    protocol.update(
        models=list(models),
        seeds=[6, 7, 8, 9, 10],
        tasks=["doom-e1m1"],
        task_settings={"doom-e1m1": {"episode_timeout_seconds": 600}},
        codex_agent_version="0.155.1",
    )
    conditions = {}
    for i, model in enumerate(models):
        status = ("dead", None, "success")[i % 3]
        n = int(status is not None)
        success = int(status == "success")
        conditions[model] = {
            "doom-e1m1": {
                "expected": 5,
                "valid_natural": n,
                "unresolved_invalid": [],
                "per_seed_primary": {"6": success} if n else {},
                "seeds": {"6": status} if n else {},
                "primary_mean": success if n else None,
                "successes": success,
            }
        }
    return {
        "protocol": protocol,
        "conditions": conditions,
        "indexed_trials": 2,
        "expected_trials": 5 * len(models),
        "incomplete_retry_attempts": 0,
    }


class PaperResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = [Path(self.temporary.name) / f"{i}.json" for i in range(3)]
        self.summaries = [summary_fixture(model) for model in BASELINE_MODELS]

    def combine(self) -> dict:
        for path, summary in zip(self.paths, self.summaries):
            path.write_text(json.dumps(summary), encoding="utf-8")
        return combine_summaries(self.paths)

    def test_completed_results_render_counts_and_sample_deviation(self) -> None:
        table = render_table(self.combine())
        self.assertIn("Basic (goal) & 5/5 & 5/5 & 5/5", table)
        self.assertIn(r"$1.0 \pm 0.0$", table)
        self.assertIn(r"\shortstack{6\\Luna}", table)
        self.assertIn(r"\shortstack{6.1\\Sol}", table)
        self.assertIn(
            r"Basic (goal) & 5/5 & 5/5 & 5/5 & \pending & \pending & \pending", table
        )

    def test_baseline_still_requires_all_three_completed_models(self) -> None:
        for path, summary in zip(self.paths, self.summaries):
            path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires all three GPT-5.6"):
            combine_summaries(self.paths[:2])

    def test_legacy_baseline_records_reconstructed_controls_explicitly(self) -> None:
        for summary in self.summaries:
            del summary["protocol"]["task_settings"]
        data = self.combine()
        self.assertEqual(data["standard_task_settings"], LEGACY_TASK_SETTINGS)
        self.assertTrue(
            all(source["legacy_task_controls"] for source in data["sources"])
        )

    def additional_fixture(self, completed: int = 1) -> dict:
        summary = summary_fixture(MODELS[3])
        summary["indexed_trials"] = completed * len(TASKS)
        summary["protocol"]["codex_agent_version"] = "new-cli-version"
        summary["protocol"]["source_sha256"] = {"server.py": "streaming-revision"}
        for task, result in summary["conditions"][MODELS[3]].items():
            result.update(
                valid_natural=completed,
                per_seed_primary={str(seed): 1 for seed in range(1, completed + 1)},
                seeds={str(seed): "success" for seed in range(1, completed + 1)},
                primary_mean=1 if completed else None,
                successes=completed if task in ("basic", "my-way-home") else None,
            )
        return summary

    def add_additional(self, data: dict, fixture: dict) -> None:
        path = Path(self.temporary.name) / "additional.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        add_additional_summaries(data, [path])

    def test_partial_new_model_standard_outcomes_are_explicit(self) -> None:
        data = self.combine()
        self.add_additional(data, self.additional_fixture())
        self.add_levels(data)
        table = render_table(data)
        self.assertIn("Basic (goal) & 5/5 & 5/5 & 5/5 & 1/1 [1/5]", table)
        self.assertIn(r"$1.0$ [1/5]", table)
        self.assertIn(r"\newcommand{\standardEpisodeCount}{64}", render_status(data))
        source = data["condition_sources"][MODELS[3]]["basic"]
        self.assertEqual(source["codex_agent_version"], "new-cli-version")
        self.assertEqual(source["source_sha256"]["server.py"], "streaming-revision")

    def test_registered_new_model_without_outcomes_is_not_zero_success(self) -> None:
        data = self.combine()
        self.add_additional(data, self.additional_fixture(0))
        self.assertIn(
            r"Basic (goal) & 5/5 & 5/5 & 5/5 & \pending{} [0/5]", render_table(data)
        )

    def test_additional_standard_seeds_must_match_baseline(self) -> None:
        fixture = self.additional_fixture()
        fixture["protocol"]["seeds"][-1] = 6
        with self.assertRaisesRegex(ValueError, "Unmatched additional control"):
            self.add_additional(self.combine(), fixture)

    def test_additional_standard_task_controls_must_match_baseline(self) -> None:
        fixture = self.additional_fixture()
        fixture["protocol"]["task_settings"]["basic"]["max_decisions"] = 150
        with self.assertRaisesRegex(ValueError, "Unmatched additional task settings"):
            self.add_additional(self.combine(), fixture)

    def test_duplicate_standard_condition_is_rejected(self) -> None:
        data = self.combine()
        fixture = self.additional_fixture()
        self.add_additional(data, fixture)
        with self.assertRaisesRegex(ValueError, "Duplicate standard condition"):
            self.add_additional(data, fixture)

    def test_different_seed_suite_is_rejected(self) -> None:
        self.summaries[-1]["protocol"]["seeds"][-1] = 6
        with self.assertRaisesRegex(ValueError, "Unmatched seeds"):
            self.combine()

    def test_unfinished_episode_is_rejected(self) -> None:
        self.summaries[-1]["conditions"][BASELINE_MODELS[-1]]["basic"][
            "valid_natural"
        ] = 4
        with self.assertRaisesRegex(ValueError, "Incomplete natural endings"):
            self.combine()

    def test_stale_original_status_is_rejected(self) -> None:
        self.summaries[-1]["conditions"][BASELINE_MODELS[-1]]["basic"]["seeds"][
            "1"
        ] = "agent_stopped"
        with self.assertRaisesRegex(ValueError, "Non-terminal seed outcomes"):
            self.combine()

    def add_levels(self, data: dict, fixture: dict | None = None) -> None:
        path = Path(self.temporary.name) / "level.json"
        path.write_text(json.dumps(fixture or level_fixture()), encoding="utf-8")
        add_level_summaries(data, [path])

    def test_partial_levels_do_not_count_pending_trials_as_failures(self) -> None:
        data = self.combine()
        self.add_levels(data)
        table = render_table(data)
        self.assertIn("Basic (goal) & 5/5 & 5/5 & 5/5", table)
        self.assertIn(
            r"Hangar (exit) & 0/1 [1/5] & \pending{} [0/5] & 1/1 [1/5]", table
        )
        self.assertIn(r"Entryway (exit) & \pending & \pending & \pending", table)
        self.assertIn(r"\newcommand{\standardEpisodeCount}{60}", render_status(data))
        self.assertIn(r"\newcommand{\levelEpisodeCount}{2}", render_status(data))
        self.assertIn(r"\newcommand{\levelExpectedCount}{15}", render_status(data))
        self.assertNotEqual(
            data["level_settings"]["doom-e1m1"]["seeds"],
            data["matched_settings"]["seeds"],
        )

    def test_nonterminal_level_outcome_is_rejected(self) -> None:
        fixture = level_fixture()
        fixture["conditions"][MODELS[0]]["doom-e1m1"]["seeds"]["6"] = "running"
        with self.assertRaisesRegex(ValueError, "Non-terminal outcome"):
            self.add_levels(self.combine(), fixture)

    def test_incorrect_level_sample_count_is_rejected(self) -> None:
        fixture = level_fixture()
        fixture["conditions"][MODELS[0]]["doom-e1m1"]["valid_natural"] = 5
        with self.assertRaisesRegex(ValueError, "Invalid completed sample"):
            self.add_levels(self.combine(), fixture)

    def test_different_level_controls_are_rejected(self) -> None:
        fixture = level_fixture()
        fixture["protocol"]["frame_skip"] = 8
        with self.assertRaisesRegex(ValueError, "Unmatched level control setting"):
            self.add_levels(self.combine(), fixture)

    def test_level_status_and_success_must_agree(self) -> None:
        fixture = level_fixture()
        fixture["conditions"][MODELS[0]]["doom-e1m1"]["seeds"]["6"] = "success"
        with self.assertRaisesRegex(ValueError, "Invalid success counts"):
            self.add_levels(self.combine(), fixture)

    def test_same_level_cannot_mix_different_paired_seed_suites(self) -> None:
        fixtures = [level_fixture(), level_fixture()]
        fixtures[1]["protocol"]["seeds"][-1] = 11
        paths = []
        for i, fixture in enumerate(fixtures):
            path = Path(self.temporary.name) / f"level-{i}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            paths.append(path)
        with self.assertRaisesRegex(ValueError, "Unmatched paired level settings"):
            add_level_summaries(self.combine(), paths)

    def test_completed_levels_drop_partial_marker(self) -> None:
        data = self.combine()
        fixture = level_fixture()
        fixture["indexed_trials"] = 15
        for tasks in fixture["conditions"].values():
            result = tasks["doom-e1m1"]
            result.update(
                valid_natural=5,
                successes=1,
                primary_mean=0.2,
                seeds={
                    str(seed): "success" if seed == 6 else "dead"
                    for seed in fixture["protocol"]["seeds"]
                },
                per_seed_primary={
                    str(seed): int(seed == 6) for seed in fixture["protocol"]["seeds"]
                },
            )
        self.add_levels(data, fixture)
        self.assertIn("Hangar (exit) & 1/5 & 1/5 & 1/5", render_table(data))

    def test_new_level_models_can_record_different_cli_and_sources(self) -> None:
        fixtures = [level_fixture(), level_fixture(models=MODELS[3:])]
        fixtures[1]["protocol"]["codex_agent_version"] = "updated-cli"
        fixtures[1]["protocol"]["source_sha256"] = {"server.py": "streaming"}
        paths = []
        for i, fixture in enumerate(fixtures):
            path = Path(self.temporary.name) / f"level-{i}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            paths.append(path)
        data = self.combine()
        add_level_summaries(data, paths)
        self.assertEqual(
            data["level_condition_sources"][MODELS[3]]["doom-e1m1"][
                "codex_agent_version"
            ],
            "updated-cli",
        )
        self.assertIn(
            r"Hangar (exit) & 0/1 [1/5] & \pending{} [0/5] & 1/1 [1/5]"
            r" & 0/1 [1/5] & \pending{} [0/5] & 1/1 [1/5]",
            render_table(data),
        )

    def test_same_level_rejects_changed_game_time_cap(self) -> None:
        fixtures = [level_fixture(), level_fixture(models=MODELS[3:])]
        fixtures[1]["protocol"]["task_settings"]["doom-e1m1"][
            "episode_timeout_seconds"
        ] = 1200
        paths = []
        for i, fixture in enumerate(fixtures):
            path = Path(self.temporary.name) / f"level-{i}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            paths.append(path)
        with self.assertRaisesRegex(ValueError, "Unmatched paired level settings"):
            add_level_summaries(self.combine(), paths)


if __name__ == "__main__":
    unittest.main()
