# ViZDoom Harbor evaluation

This experiment compares Codex and Claude Code on the first two maps of each
commercial game:

| Task | Gymnasium environment | Level |
| --- | --- | --- |
| DOOM E1M1 | `VizdoomDoomE1M1-S1-v0` | Hangar |
| DOOM E1M2 | `VizdoomDoomE1M2-S1-v0` | Nuclear Plant |
| DOOM II MAP01 | `VizdoomDoom2MAP01-S1-v0` | Entryway |
| DOOM II MAP02 | `VizdoomDoom2MAP02-S1-v0` | Underhalls |

Each map is an independent Harbor task at skill level 1. The game runs in a
Docker Compose sidecar and is exposed to the agent as a FastMCP server. The
Gymnasium action space is `MultiBinary`, frame skip is 4, and success requires
the environment's map-exit reward. Death, timeout, or reaching 1,000 decisions
is failure.

The MCP server exposes four tools:

- `get_current_state()` returns metadata, the first-person screen, and automap.
- `make_action(buttons)` presses zero or more named buttons and returns the new
  metadata, first-person screen, and automap.
- `store_state(memory_id)` saves the current visual state under an agent-chosen
  ID, replacing an existing state with the same ID.
- `get_stored_state(memory_id)` returns the metadata, screen, and automap for
  that explicitly stored state.

There is no tool for arbitrary previous-state access. State and automap images
are encoded as PNG bytes in memory and returned as MCP image content; no frame
PNG files are written. The sidecar records the run using Gymnasium's
`RecordVideo`, appends tool/game events to `events.jsonl`, and atomically saves
the latest outcome to `result.json`. Harbor's collection hook finalizes the
video even when the agent stops early. Harbor collects `recording.mp4`,
`events.jsonl`, and `result.json` from the sidecar for every trial.

## Requirements

- Docker with Docker Compose
- [Harbor](https://www.harborframework.com/docs/getting-started/installation)
- A legally obtained `doom.wad` and `doom2.wad`
- `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` for the default run config

The commercial IWADs are not part of this repository or copied into an image.
They are mounted read-only into their respective sidecars. Set their absolute
host paths:

```bash
export DOOM_WAD_PATH=/absolute/path/to/doom.wad
export DOOM2_WAD_PATH=/absolute/path/to/doom2.wad
```

Install Harbor, if needed:

```bash
uv tool install harbor
```

## Run

From the ViZDoom repository root:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python experiments/harbor_vizdoom/run_experiment.py
```

The launcher generates one random unsigned 32-bit seed and shares it across
all maps and agents in that run. Supply `--seed` for reproducibility:

```bash
python experiments/harbor_vizdoom/run_experiment.py --seed 124343
```

Run a single agent or change its model/reasoning effort with, for example:

```bash
python experiments/harbor_vizdoom/run_experiment.py \
  --agent codex \
  --codex-model openai/gpt-5.6-luna \
  --codex-reasoning high
```

The output directory includes the selected agent/model, reasoning effort, and
actual seed:

```text
jobs/
  models-codex-openai_gpt-5.6-luna/
    reasoning-codex-high/
      seed-124343/
```

Use `--dry-run` to inspect the generated Harbor config without checking IWADs
or starting Harbor. `run.yaml` is also available as a simpler static config;
when using it directly, set `VIZDOOM_SEED` yourself.

## Isolation and outputs

Codex receives `web_search: disabled`; Claude Code receives
`disallowed_tools: WebSearch,WebFetch`. The agent phase still requires network
access to its model provider, so this is not a general network sandbox. Task
instructions also prohibit internet access through shell commands.

Each trial contains the agent trajectory, verifier reward, and collected
`vizdoom-results/result.json`, `vizdoom-results/events.jsonl`, and
`vizdoom-results/recording.mp4` artifacts.
The Compose MCP sidecar requires Harbor's local Docker environment provider.

ViZDoom's repository documentation for these environments is in
[`docs/environments/original_doom_levels.md`](../../docs/environments/original_doom_levels.md).
