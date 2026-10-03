"""Memory supervision must preserve at least one episode and isolate its scope."""

import argparse
import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from supervise_pilot import (
    _jsonl,
    _log,
    _memory_mib,
    _owned_containers,
    _pause_plan,
    _resume_owned_pauses,
    _stdout,
    _supervise,
)


class PilotSupervisorTests(unittest.TestCase):
    def test_pressure_keeps_furthest_progressed_episode(self):
        agents = {name: {"Paused": False} for name in ("a", "b", "c")}
        conserving, pause, resume = _pause_plan(
            agents, set(), {"a": 200, "b": 400, "c": 100}, 1500, False, 2048, 3584
        )
        self.assertTrue(conserving)
        self.assertEqual(pause, {"a", "c"})
        self.assertEqual(resume, set())

    def test_hysteresis_keeps_paused_agents_until_memory_recovers(self):
        agents = {"a": {"Paused": True}, "b": {"Paused": False}}
        args = (agents, {"a"}, {"a": 200, "b": 400})
        self.assertEqual(
            _pause_plan(*args, 3000, True, 2048, 3584), (True, set(), set())
        )
        self.assertEqual(
            _pause_plan(*args, 4000, True, 2048, 3584), (False, set(), {"a"})
        )

    def test_finished_priority_trial_releases_a_paused_trial(self):
        agents = {
            "a": {"Paused": True},
            "b": {"Paused": True},
            "new": {"Paused": False},
        }
        self.assertEqual(
            _pause_plan(
                agents, {"a", "b"}, {"a": 200, "b": 300}, 2000, True, 2048, 3584
            ),
            (True, {"new"}, {"b"}),
        )

    def test_never_unpauses_a_container_paused_by_someone_else(self):
        self.assertEqual(
            _pause_plan(
                {"user": {"Paused": True}}, set(), {"user": 999}, 4000, True, 2048, 3584
            ),
            (False, set(), set()),
        )

    def test_discovery_only_inspects_containers_owned_by_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / "jobs/job/doom-e1m1__ABC"
            trial.mkdir(parents=True)
            (trial / "trial.log").touch()
            name = "doom-e1m1__abc__env-main-1"
            info = {"Name": "/" + name, "State": {"Running": True, "Paused": False}}
            with patch(
                "supervise_pilot._docker",
                side_effect=[name + "\nunrelated__env-main-1\n", json.dumps([info])],
            ) as docker:
                self.assertEqual(_owned_containers(root), {name: info})
            self.assertEqual(docker.call_args.args, ("inspect", name))

    def test_jsonl_ignores_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs.jsonl"
            path.write_text('{"seed": 1}\n{"seed":', encoding="utf-8")
            self.assertEqual(_jsonl(path), [{"seed": 1}])

    def test_memory_unit_conversion(self):
        self.assertEqual(_memory_mib("MemTotal: 999 kB\nMemAvailable: 2048 kB\n"), 2)

    def test_broken_stdout_preserves_durable_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("builtins.print", side_effect=BrokenPipeError()):
                _log(root, "status", indexed=4)
            events = _jsonl(root / "supervision.jsonl")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "status")
            self.assertEqual(events[0]["indexed"], 4)

    def test_stdout_only_ignores_broken_pipe(self):
        with patch("builtins.print", side_effect=OSError(errno.EPIPE, "pipe closed")):
            _stdout("status")
        with patch("builtins.print", side_effect=OSError(errno.EIO, "I/O failure")):
            with self.assertRaises(OSError) as raised:
                _stdout("status")
        self.assertEqual(raised.exception.errno, errno.EIO)

    def test_supervisor_cleanup_resumes_every_pause_with_broken_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(
                json.dumps({"models": ["model"], "seeds": [1], "tasks": ["basic"]}),
                encoding="utf-8",
            )
            with patch("builtins.print"):
                _log(root, "pause", container="a")
                _log(root, "pause", container="b")
            args = argparse.Namespace(
                manage_memory=True, reserve_mib=2048, resume_mib=3584, poll_seconds=30
            )
            with (
                patch("builtins.print", side_effect=BrokenPipeError()),
                patch("supervise_pilot._docker", return_value="") as docker,
                patch("supervise_pilot._run_helper", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                _supervise(args, root)
            self.assertEqual(
                docker.call_args_list, [call("unpause", "a"), call("unpause", "b")]
            )
            resumed = [
                e["container"]
                for e in _jsonl(root / "supervision.jsonl")
                if e["event"] == "resume"
            ]
            self.assertEqual(resumed, ["a", "b"])

    def test_cleanup_reports_logfile_errors_after_all_unpause_attempts(self):
        disk_error = OSError(errno.ENOSPC, "disk full")
        with (
            patch("supervise_pilot._docker", return_value="") as docker,
            patch("supervise_pilot._log", side_effect=disk_error),
            self.assertRaises(OSError) as raised,
        ):
            _resume_owned_pauses(Path("unused"), {"a", "b"})
        self.assertIs(raised.exception, disk_error)
        self.assertEqual(
            docker.call_args_list, [call("unpause", "a"), call("unpause", "b")]
        )

    def test_failed_unpause_is_logged_without_blocking_other_containers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = subprocess.CalledProcessError(1, ["docker", "unpause", "a"])
            with (
                patch("supervise_pilot._docker", side_effect=[failure, ""]) as docker,
                patch("builtins.print", side_effect=BrokenPipeError()),
            ):
                _resume_owned_pauses(root, {"a", "b"})
            self.assertEqual(
                docker.call_args_list, [call("unpause", "a"), call("unpause", "b")]
            )
            events = _jsonl(root / "supervision.jsonl")
            self.assertEqual(
                [(e["event"], e["container"]) for e in events],
                [("cleanup_warning", "a"), ("resume", "b")],
            )
            self.assertIn("returned non-zero exit status 1", events[0]["detail"])


if __name__ == "__main__":
    unittest.main()
