"""Helpers for checkpoint evaluation (on fixed policy)"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import Compose, TransformedEnv


Z_95 = 1.959963984540054


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a checkpoint or pick the largest numbered checkpoint in a run"""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix == ".pt":
        return path
    if path.is_dir():
        checkpoints = []
        for candidate in (path / "checkpoints").glob("checkpoint_*.pt"):
            match = re.fullmatch(r"checkpoint_(\d+)\.pt", candidate.name)
            if match:
                checkpoints.append((int(match[1]), candidate))
        if checkpoints:
            return max(checkpoints)[1]
    raise ValueError(f"No checkpoint found at {path}")


def close_env(env):
    if env is not None and not getattr(env, "is_closed", False):
        env.close()


def close_reloaded_experiment(experiment):
    # Attempt every cleanup even if one resource's close fails.
    for name in ("rollout_env", "test_env", "logger"):
        resource = getattr(experiment, name, None)
        if resource is None:
            continue
        try:
            if name == "logger":
                resource.finish()
            else:
                close_env(resource)
        except Exception as exc:
            warnings.warn(f"Failed to close reloaded {name}: {exc}", stacklevel=2)


def make_env(experiment, seed, stack, label="Checkpoint"):
    """Build one unbatched CPU task env with the training transforms."""
    task = experiment.task
    env = task.get_env_fun(
        num_envs=1,
        continuous_actions=experiment.continuous_actions,
        seed=seed,
        device="cpu",
    )()
    stack.callback(close_env, env)
    env = TransformedEnv(env, Compose(*task.get_env_transforms(env))).to("cpu")
    stack.callback(close_env, env)
    if experiment.model_config.is_rnn:
        from benchmarl.experiment.experiment import _add_rnn_transforms

        base = env
        env = _add_rnn_transforms(
            lambda: base, experiment.group_map, experiment.model_config
        )()
        stack.callback(close_env, env)
    if env.batch_size:
        raise ValueError(f"{label} evaluation requires an unbatched task environment")
    return env


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def open_video_writer(path, fps, stack):
    """Stream readable H.264 observer video. Closed by `stack`."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )
    stack.callback(writer.close)
    return writer


def wilson_interval(successes, total, z=Z_95):
    if total <= 0:
        return None, None  # without observation
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total]")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(successes, total):
    """JSON format with denominator and Wilson 95% interval."""
    low, high = wilson_interval(successes, total)
    return {
        "count": int(successes),
        "denominator": int(total),
        "rate": successes / total if total else None,
        "wilson95": [low, high],
    }


def stratified_bootstrap(strata, statistic, *, resamples=2000, seed=0, alpha=0.05):
    """Percentile CI resampling units with replacement every stratum"""
    # Map a stratum (e.g. an independent training seed) to a list of units
    strata = {key: list(units) for key, units in strata.items() if units}
    if not strata:
        return None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sample = {
            key: [units[i] for i in rng.integers(0, len(units), len(units))]
            for key, units in strata.items()
        }
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None, None
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def atomic_write_text(path, text):
    """Replace `path` only once the whole text is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def append_jsonl(path, row):
    """Append one complete JSON line and flush it (resumable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(row, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
    return rows
