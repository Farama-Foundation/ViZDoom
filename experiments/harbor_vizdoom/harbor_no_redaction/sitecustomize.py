"""Disable Harbor's destructive post-trial result redaction.

Python imports ``sitecustomize`` automatically during interpreter startup when
its directory is on ``PYTHONPATH``. The ViZDoom launcher injects this directory
only into the Harbor subprocess, so the installed Harbor package is unchanged.

This intentionally disables secret scrubbing for every textual file in a
trial directory. Do not publish a resulting job directory without inspecting
it for credentials first.
"""

from typing import Any

from harbor.trial.trial import Trial


def _skip_jobs_dir_scrubbing(_trial: Any) -> None:
    """Keep Harbor trial files byte-for-byte as originally written."""


setattr(Trial, "_scrub_jobs_dir", _skip_jobs_dir_scrubbing)
