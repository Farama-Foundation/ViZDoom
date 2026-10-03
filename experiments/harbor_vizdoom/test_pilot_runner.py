"""Regression checks for pilot job completion and log accounting."""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generate_tasks import load_task_definitions
from run_pilot import (
    BASELINE_MODELS,
    EXPERIMENT_DIR,
    FROZEN_SOURCES,
    MODELS,
    _codex_log_metrics,
    _collect_job,
    _completed_existing_job,
    _detach,
    _freeze_sources,
    _index_records,
    _protocol_tasks,
    _reuse_records,
    _run_job,
    _run_pending,
    _source_root,
    _status,
)
from summarize_pilot import _valid_trial
from summarize_pilot import main as summarize


class PilotRunnerTests(unittest.TestCase):
    protocol = {
        "reasoning": "medium",
        "auth": "subscription",
        "concurrent_trials": 2,
    }
    model = "openai/gpt-5.6-luna"
    seed = 12345

    def test_new_model_choices_keep_default_baseline(self) -> None:
        self.assertEqual(len(BASELINE_MODELS), 3)
        self.assertEqual(len(MODELS), 6)
        self.assertIn("openai/gpt-6.1-sol", MODELS)

    def test_archived_standard_sources_include_empty_iwad_context_and_doom_links(self):
        protocol = {
            "source_sha256": {
                source: hashlib.sha256(
                    (EXPERIMENT_DIR / source).read_bytes()
                ).hexdigest()
                for source in FROZEN_SOURCES
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _freeze_sources(run_dir)
            with tarfile.open(run_dir / "source_snapshot.tar.gz") as archive:
                names = archive.getnames()
            self.assertFalse(
                any(name.endswith((".wad", "auth.json")) for name in names)
            )
            root = _source_root(run_dir, protocol)
            self.assertTrue(
                (root / "template/common/environment/vizdoom-server/iwads").is_dir()
            )
            self.assertEqual(
                len(
                    load_task_definitions(root / "environments.json", root / "template")
                ),
                8,
            )
            self.assertTrue((root / "template/doom.wad").is_symlink())

    def test_broken_stdout_does_not_stop_indexing(self) -> None:
        record = {
            "model": self.model,
            "seed": self.seed,
            "task": "basic",
            "status": "dead",
            "natural_end": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs.jsonl"
            indexed = set()
            with patch("builtins.print", side_effect=BrokenPipeError):
                _status("closed client")
                _index_records(path, indexed, [record])
            self.assertEqual(json.loads(path.read_text()), record)
            self.assertEqual(indexed, {(self.model, self.seed, "basic")})

    def test_detached_processes_have_file_output_and_no_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            args = SimpleNamespace(parallel_jobs=2, skip_in_progress=False)
            with patch(
                "run_pilot.subprocess.Popen", return_value=SimpleNamespace(pid=54321)
            ) as launch:
                _detach(run_dir, args)
            self.assertEqual(launch.call_count, 2)
            for call in launch.call_args_list:
                self.assertTrue(call.kwargs["start_new_session"])
                self.assertEqual(call.kwargs["stdin"], -3)
                self.assertNotEqual(call.kwargs["stdout"], -1)
            self.assertEqual(
                json.loads((run_dir / "launcher.json").read_text())["runner"]["pid"],
                54321,
            )

    def test_reused_natural_failure_is_not_replayed_or_improved(self) -> None:
        task = {
            "environment_id": "test-doom",
            "episode_timeout_seconds": 600,
            "max_decisions": 5250,
            "require_automap": True,
        }
        protocol = {
            **self.protocol,
            "models": [self.model],
            "tasks": ["doom-e1m1"],
            "seeds": [self.seed],
            "frame_skip": 4,
            "max_action_repeat": 2,
            "task_settings": {"doom-e1m1": task},
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "protocol.json").write_text(json.dumps(protocol))
            record = {
                "model": self.model,
                "seed": self.seed,
                "task": "doom-e1m1",
                "status": "dead",
                "success": False,
                "trial_dir": str(source / "trial"),
            }
            (source / "runs.jsonl").write_text(json.dumps(record) + "\n")
            with patch("summarize_pilot._valid_trial", return_value=True):
                reused = _reuse_records(protocol, [source])
                self.assertEqual(reused[0]["status"], "dead")
                self.assertFalse(reused[0]["success"])
                self.assertEqual(len(protocol["reused_outcomes"]), 1)
                changed = {
                    **protocol,
                    "task_settings": {
                        "doom-e1m1": {**task, "episode_timeout_seconds": 1200}
                    },
                }
                with self.assertRaisesRegex(ValueError, "gameplay settings"):
                    _reuse_records(changed, [source])

    def test_only_missing_task_cells_are_launched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            indexed = {(self.model, self.seed, "doom-e1m1")}
            protocol = {**self.protocol, "tasks": ["doom-e1m1", "doom-e1m2"]}
            with patch("run_pilot._run_job", return_value=[]) as launch:
                _run_pending(
                    run_dir,
                    protocol,
                    run_dir,
                    [(self.model, self.seed)],
                    run_dir / "runs.jsonl",
                    indexed,
                    1,
                )
            self.assertEqual(launch.call_args.args[1]["tasks"], ["doom-e1m2"])

    def test_worker_failure_preserves_prior_results_and_stops_queue(self) -> None:
        record = {
            "model": self.model,
            "seed": 1,
            "task": "basic",
            "status": "dead",
            "natural_end": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            path = run_dir / "runs.jsonl"
            indexed = set()
            with patch(
                "run_pilot._run_job",
                side_effect=[[record], RuntimeError("daemon lost")],
            ) as launch:
                with self.assertRaisesRegex(RuntimeError, "preserved completed"):
                    _run_pending(
                        run_dir,
                        {**self.protocol, "tasks": ["basic"]},
                        run_dir,
                        [(self.model, seed) for seed in (1, 2, 3)],
                        path,
                        indexed,
                        1,
                    )
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(json.loads(path.read_text())["seed"], 1)
            self.assertEqual(
                json.loads((run_dir / "launch_failures.jsonl").read_text())["seed"], 2
            )

    def test_harbor_errors_do_not_double_count_completed_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            job_root = (
                run_dir
                / "jobs/models-codex-openai_gpt-5.6-luna"
                / "reasoning-codex-medium/auth-codex-subscription/seed-12345"
            )
            job_dir = job_root / "vizdoom-20260922T000000Z"
            job_dir.mkdir(parents=True)
            result_path = job_dir / "result.json"
            result = {
                "finished_at": "2026-09-22T00:00:00Z",
                "stats": {"n_completed_trials": 3, "n_errored_trials": 1},
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertFalse(
                _completed_existing_job(run_dir, self.protocol, self.model, self.seed)
            )
            result["stats"]["n_completed_trials"] = 4
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertTrue(
                _completed_existing_job(run_dir, self.protocol, self.model, self.seed)
            )

    def test_setup_failure_is_not_indexed_as_four_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir()
            with patch(
                "run_pilot.subprocess.run", return_value=SimpleNamespace(returncode=1)
            ):
                with self.assertRaisesRegex(RuntimeError, "No episode records"):
                    _run_job(
                        run_dir,
                        self.protocol,
                        logs_dir,
                        self.model,
                        self.seed,
                    )

    def test_selected_task_controls_launch_and_completion(self) -> None:
        protocol = {**self.protocol, "tasks": ["doom-e1m1"]}
        self.assertEqual(_protocol_tasks(protocol), ("doom-e1m1",))
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir()
            job_dir = (
                run_dir
                / "jobs/models-codex-openai_gpt-5.6-luna"
                / "reasoning-codex-medium/auth-codex-subscription/seed-12345"
                / "vizdoom-20260925T000000Z"
            )
            job_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text(
                json.dumps({"finished_at": "now", "stats": {"n_completed_trials": 1}}),
                encoding="utf-8",
            )
            self.assertTrue(
                _completed_existing_job(run_dir, protocol, self.model, self.seed)
            )
            with patch(
                "run_pilot.subprocess.run", return_value=SimpleNamespace(returncode=0)
            ) as launch, patch("run_pilot._collect_job", return_value=[]) as collect:
                _run_job(run_dir, protocol, logs_dir, self.model, self.seed)
            command = launch.call_args.args[0]
            self.assertEqual(command.count("--task"), 1)
            self.assertEqual(command[-2:], ["--task", "doom-e1m1"])
            self.assertEqual(collect.call_args.args[1]["tasks"], ["doom-e1m1"])

    def test_codex_log_counts_turns_and_image_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "codex.txt"
            path.write_text(
                "warning before JSON\n"
                '{"type":"item.completed","item":{"type":"image"}}\n'
                '{"type":"turn.completed"}\n'
                '{"type":"turn.completed"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                _codex_log_metrics(path),
                {"cli_turns": 2, "image_tool_results": 1},
            )

    def test_doom_summary_counts_only_selected_goal_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            protocol = {
                **self.protocol,
                "models": [self.model],
                "tasks": ["doom-e1m1"],
                "seeds": [self.seed],
            }
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            record = {
                "model": self.model,
                "seed": self.seed,
                "task": "doom-e1m1",
                "trial_dir": str(run_dir / "trial"),
                "success": True,
                "decisions": 10,
                "status": "success",
            }
            (run_dir / "runs.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            with patch("sys.argv", ["summarize_pilot.py", str(run_dir)]), patch(
                "summarize_pilot._valid_trial", return_value=True
            ):
                summarize()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["expected_trials"], 1)
            condition = summary["conditions"][self.model]["doom-e1m1"]
            self.assertEqual(condition["primary_measure"], "success_rate")
            self.assertEqual(condition["successes"], 1)

    def test_score_requires_image_bearing_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial_dir = Path(temporary)
            (trial_dir / "agent").mkdir()
            (trial_dir / "result.json").write_text(
                json.dumps({"exception_info": None}), encoding="utf-8"
            )
            codex_log = trial_dir / "agent/codex.txt"
            codex_log.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
            record = {
                "trial_dir": str(trial_dir),
                "artifacts": {
                    "result.json": True,
                    "events.jsonl": True,
                    "recording.mp4": True,
                },
                "trajectory_present": True,
                "verifier_reward": 1.0,
                "status": "success",
            }
            self.assertFalse(_valid_trial(record, "basic"))
            codex_log.write_text(
                '{"type":"item.completed","item":{"type":"image"}}\n',
                encoding="utf-8",
            )
            self.assertTrue(_valid_trial(record, "basic"))

    def _live_trial(self, root: Path, protocol: dict, seed: int, finished=True):
        trial = (
            root
            / "jobs/models-codex-openai_gpt-5.6-luna"
            / "reasoning-codex-medium/auth-codex-subscription"
            / f"seed-{seed}/vizdoom-20261001T000000Z/doom-e1m1__ABC"
        )
        artifacts = trial / "artifacts/vizdoom-results"
        artifacts.mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "verifier").mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "finished_at": "2026-10-01T00:00:00Z" if finished else None,
                    "exception_info": None,
                }
            )
        )
        (artifacts / "result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "success": True,
                    "decisions": 10,
                    "episode_return": 1.0,
                }
            )
        )
        (artifacts / "events.jsonl").write_text('{"event":"finalize"}\n')
        (artifacts / "recording.mp4").write_bytes(b"video fixture")
        (trial / "agent/codex.txt").write_text(
            '{"type":"item.completed","item":{"type":"image"}}\n'
            '{"type":"turn.completed"}\n'
        )
        (trial / "agent/trajectory.json").write_text(
            json.dumps({"agent": {"version": "0.155.1"}, "steps": []})
        )
        (trial / "verifier/metrics.json").write_text(
            json.dumps({"reward": 1.0, "narration_coverage": 1.0})
        )
        return trial

    def test_live_summary_includes_finished_valid_trials_without_changing_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {
                **self.protocol,
                "models": [self.model],
                "tasks": ["doom-e1m1"],
                "seeds": [1, 2, 3, 4, 5],
                "reused_outcomes": [{"seed": 1, "run_dir": "original-run"}],
            }
            (root / "protocol.json").write_text(json.dumps(protocol))
            self._live_trial(root, protocol, 1)
            discovered_trial = self._live_trial(root, protocol, 2)
            self._live_trial(root, protocol, 3, finished=False)
            invalid = self._live_trial(root, protocol, 4)
            (invalid / "verifier/metrics.json").write_text("{}")
            transient = self._live_trial(root, protocol, 5)
            (transient / "artifacts/vizdoom-results/events.jsonl").write_text(
                '{"event":'
            )
            indexed_record = _collect_job(root, protocol, self.model, 1, 0)[0]
            indexed_record["reused_from"] = {"run_dir": "original-run"}
            original_index = json.dumps(indexed_record) + "\n"
            (root / "runs.jsonl").write_text(original_index)
            original_summary = '{"existing_summary": true}\n'
            (root / "summary.json").write_text(original_summary)
            with patch(
                "sys.argv",
                ["summarize_pilot.py", str(root), "--include-completed-trials"],
            ), patch("builtins.print"):
                summarize()
            self.assertEqual((root / "runs.jsonl").read_text(), original_index)
            self.assertEqual((root / "summary.json").read_text(), original_summary)
            summary = json.loads((root / "live_summary.json").read_text())
            self.assertEqual(summary["protocol"], protocol)
            self.assertEqual(summary["snapshot_mode"], "read_only_completed_trials")
            self.assertEqual(summary["journal_indexed_trials"], 1)
            self.assertEqual(summary["discovered_completed_trials"], 1)
            self.assertEqual(summary["indexed_trials"], 2)
            self.assertEqual(
                summary["conditions"][self.model]["doom-e1m1"]["valid_natural"], 2
            )
            record = summary["discovered_completed_records"][0]
            self.assertEqual(record["seed"], 2)
            self.assertEqual(Path(record["trial_dir"]), discovered_trial.resolve())
            self.assertEqual(record["trial_finished_at"], "2026-10-01T00:00:00Z")
            self.assertIsNone(record["harbor_returncode"])
            self.assertEqual(
                {candidate["seed"] for candidate in summary["skipped_live_candidates"]},
                {4, 5},
            )

    def test_default_summary_does_not_discover_finished_unindexed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {
                **self.protocol,
                "models": [self.model],
                "tasks": ["doom-e1m1"],
                "seeds": [1],
            }
            (root / "protocol.json").write_text(json.dumps(protocol))
            self._live_trial(root, protocol, 1)
            with patch("sys.argv", ["summarize_pilot.py", str(root)]), patch(
                "builtins.print"
            ):
                summarize()
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["indexed_trials"], 0)
            self.assertNotIn("snapshot_mode", summary)
            self.assertFalse((root / "runs.jsonl").exists())
            self.assertFalse((root / "live_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
