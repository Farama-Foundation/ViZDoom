"""Per-run sources must be selected without changing process-wide defaults."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_tasks as generator
import run_experiment as runner


class SourceSnapshotTests(unittest.TestCase):
    def _snapshot(self, parent: Path) -> Path:
        root = parent / "snapshot"
        root.mkdir()
        shutil.copytree(
            generator.TEMPLATE_DIR,
            root / "template",
            ignore=shutil.ignore_patterns("*.wad", "__pycache__"),
        )
        for name in ("doom.wad", "doom2.wad"):
            (root / "template" / name).symlink_to(generator.TEMPLATE_DIR / name)
        shutil.copytree(
            runner.HARBOR_NO_REDACTION_DIR,
            root / "harbor_no_redaction",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        task = next(
            t for t in generator.load_task_definitions() if t["slug"] == "basic"
        )
        task["display_name"] = "Archived Basic"
        (root / "environments.json").write_text(
            json.dumps({"schema_version": 2, "tasks": [task]}), encoding="utf-8"
        )
        for relative in (
            "task.toml.template",
            "docker-compose.yaml.template",
            "run.yaml.template",
            "instructions/basic.md.template",
            "common/environment/vizdoom-server/server.py",
        ):
            path = root / "template" / relative
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n# archived source marker\n")
        return root.resolve()

    def test_generator_uses_snapshot_manifest_and_every_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = self._snapshot(parent)
            manifest = root / "environments.json"
            templates = root / "template"
            definitions = generator.load_task_definitions(manifest, templates)
            self.assertEqual(
                [t["display_name"] for t in definitions], ["Archived Basic"]
            )
            task = generator.generate_tasks(
                parent / "tasks", ["basic"], manifest, templates
            )[0]
            for relative in (
                "task.toml",
                "environment/docker-compose.yaml",
                "instruction.md",
                "environment/vizdoom-server/server.py",
            ):
                self.assertIn(
                    "archived source marker",
                    (task / relative).read_text(encoding="utf-8"),
                )
            self.assertIn("Archived Basic", (task / "task.toml").read_text())
            self.assertIn(
                "archived source marker", (parent / "tasks/run.yaml").read_text()
            )
            self.assertNotEqual(
                next(
                    t for t in generator.load_task_definitions() if t["slug"] == "basic"
                )["display_name"],
                "Archived Basic",
            )

    def test_runner_selects_snapshot_harness_and_preserves_auth_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = self._snapshot(parent)
            observations = []

            def launch(command, **kwargs):
                config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                task = Path(config["tasks"][0]["path"])
                observations.append(
                    (config, kwargs["env"], (task / "instruction.md").read_text())
                )
                return subprocess.CompletedProcess(command, 0)

            arguments = [
                "run_experiment.py",
                "--source-root",
                str(root),
                "--agent",
                "codex",
                "--agent-auth",
                "subscription",
                "--seed",
                "42",
                "--task",
                "basic",
                "--jobs-root",
                str(parent / "jobs"),
                "--skip-docker-preflight",
            ]
            with (
                patch("sys.argv", arguments),
                patch.dict(os.environ, {"PYTHONPATH": "previous-source"}),
                patch("run_experiment.subprocess.run", side_effect=launch) as process,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.main()
            self.assertEqual(process.call_count, 1)
            config, environment, instruction = observations[0]
            self.assertEqual(
                environment["PYTHONPATH"].split(os.pathsep),
                [str(root / "harbor_no_redaction"), "previous-source"],
            )
            self.assertEqual(environment["VIZDOOM_SEED"], "42")
            self.assertEqual(config["agents"][0]["env"], {"CODEX_FORCE_AUTH_JSON": "1"})
            self.assertIn("archived source marker", instruction)

    def test_missing_snapshot_source_has_explicit_path_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._snapshot(Path(temporary))
            missing = root / "harbor_no_redaction/sitecustomize.py"
            missing.unlink()
            with self.assertRaises(SystemExit) as raised:
                runner._source_paths(root)
            self.assertIn(str(missing), str(raised.exception))

    def test_missing_snapshot_template_does_not_fall_back_to_live_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = self._snapshot(parent)
            missing = root / "template/run.yaml.template"
            missing.unlink()
            with self.assertRaises(RuntimeError) as raised:
                generator.generate_tasks(
                    parent / "tasks",
                    ["basic"],
                    root / "environments.json",
                    root / "template",
                )
            self.assertIn(str(missing), str(raised.exception))
            self.assertFalse((parent / "tasks").exists())

    def test_doom_iwad_validation_uses_snapshot_asset_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._snapshot(Path(temporary))
            task = next(
                t for t in generator.load_task_definitions() if t["slug"] == "doom-e1m1"
            )
            manifest = root / "environments.json"
            manifest.write_text(
                json.dumps({"schema_version": 2, "tasks": [task]}), encoding="utf-8"
            )
            self.assertEqual(
                generator.load_task_definitions(manifest, root / "template")[0]["slug"],
                "doom-e1m1",
            )
            missing = root / "template/doom.wad"
            missing.unlink()
            with self.assertRaises(RuntimeError) as raised:
                generator.load_task_definitions(manifest, root / "template")
            self.assertIn(str(missing), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
