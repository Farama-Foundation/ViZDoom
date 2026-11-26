#!/usr/bin/env python3

# This test should be run manually.
# Full self-check of the vizdoom-scenario-design skill: builds the probe WADs and runs
# every validate_scenario.py level on them, including 2-player netgames (ports 47700-47760).

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SKILL = Path(__file__).resolve().parents[1] / "skills" / "vizdoom-scenario-design"


def test_probe_check():
    with TemporaryDirectory() as out:
        result = subprocess.run(
            [
                sys.executable,
                str(SKILL / "scripts" / "build_probe_wads.py"),
                "--out",
                out,
                "--check",
            ],
            capture_output=True,
            text=True,
        )
    summary = [
        line for line in result.stdout.splitlines() if line.startswith(("OK ", "BAD"))
    ]
    print("\n".join(summary))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


if __name__ == "__main__":
    test_probe_check()
