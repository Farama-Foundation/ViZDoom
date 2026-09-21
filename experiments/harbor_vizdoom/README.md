# ViZDoom Harbor evaluation

This experiment generates self-contained Harbor tasks from a single template
and a manifest. The included task definitions are:

| Slug | Gymnasium environment | Objective |
| --- | --- | --- |
| `basic` | `VizdoomBasic-MultiBinary-v1` | Kill the monster |
| `my-way-home` | `VizdoomMyWayHome-MultiBinary-v1` | Reach the armor vest |
| `doom-e1m1` | `VizdoomDoomE1M1-S1-v0` | Complete Hangar |
| `doom-e1m2` | `VizdoomDoomE1M2-S1-v0` | Complete Nuclear Plant |
| `doom2-map01` | `VizdoomDoom2MAP01-S1-v0` | Complete Entryway |
| `doom2-map02` | `VizdoomDoom2MAP02-S1-v0` | Complete Underhalls |

All environments use a Gymnasium `MultiBinary` action space and frame skip 4.
Commercial DOOM maps run at skill level 1 and provide both the first-person
screen and automap. The included Basic and My Way Home scenarios return the
first-person screen only.

## Source structure

```text
harbor_vizdoom/
├── environments.json                 # Task/map definitions
├── generate_tasks.py                 # Task generator
├── harbor_no_redaction/
│   └── sitecustomize.py              # Scoped Harbor redaction monkeypatch
├── random_agent_harness.py           # Local random-policy smoke tests
├── run_experiment.py                 # Generate temporarily and run Harbor
└── template/
    ├── doom.wad                       # Bundled DOOM IWAD
    ├── doom2.wad                      # Bundled DOOM II IWAD
    ├── run.yaml.template              # Generates a static Harbor job config
    ├── task.toml.template
    ├── docker-compose.yaml.template
    ├── instructions/
    │   ├── doom.md.template           # Shared by every commercial DOOM map
    │   ├── basic.md.template
    │   └── my-way-home.md.template
    └── common/
        ├── environment/
        │   ├── Dockerfile
        │   └── vizdoom-server/
        │       ├── Dockerfile
        │       ├── iwads/             # Populated per generated task
        │       └── server.py          # The only server implementation
        └── tests/
            ├── Dockerfile
            ├── grade.py               # The only verifier implementation
            └── test.sh
```

Generated tasks are disposable, self-contained Harbor packages. They are not
committed and should not be edited. Add or change environments in
`environments.json`; change shared behavior in `template/common`; change all
commercial DOOM instructions in `template/instructions/doom.md.template`.
The generator rejects commercial DOOM definitions that select another
instruction template.

## MCP tools and results

The shared server exposes four tools:

- `get_current_state()` returns metadata and the current image buffers.
- `make_action(buttons, repeat, situation, decision)` presses zero or more
  named buttons for 1-8 repetitions and returns the new state. The two short
  narrative fields are optional in the MCP schema for compatibility, but task
  instructions require agents to provide them; they are persisted in both the
  action history and event log.
- `store_state(memory_id)` stores the current visual state under an
  agent-selected ID.
- `get_stored_state(memory_id)` returns that explicitly stored state.

There is no tool for arbitrary previous-state access. Images are encoded as
PNG bytes in memory; frame PNGs are not written. Every trial collects these
sidecar artifacts:

- `/vizdoom-results/result.json`
- `/vizdoom-results/events.jsonl`
- `/vizdoom-results/recording.mp4`

The video is recorded with Gymnasium `RecordVideo`. A Harbor collection hook
finalizes it even when the agent stops before completing the episode.

## Requirements

- Docker with Docker Compose
- [Harbor](https://www.harborframework.com/docs/getting-started/installation)
- API keys or subscription credentials for the selected agents

The `doom.wad` and `doom2.wad` files in `template/` are bundled into the
appropriate generated server build contexts. Each commercial task installs
only the IWAD it needs into `/iwads` in its sidecar image; no host IWAD path or
environment variable is required at run time. Keep these copyrighted files
private and ensure you have the right to use them.

## Run

The recommended launcher generates tasks in a temporary directory, chooses a
random unsigned 32-bit seed, and runs all six tasks:

```bash
python experiments/harbor_vizdoom/run_experiment.py
```

Select a subset by repeating `--task`:

```bash
python experiments/harbor_vizdoom/run_experiment.py \
  --task basic \
  --task my-way-home
```

List task slugs or set a reproducible seed:

```bash
python experiments/harbor_vizdoom/run_experiment.py --list-tasks
python experiments/harbor_vizdoom/run_experiment.py --seed 124343
```

The default Codex model is `openai/gpt-5.6-luna`. Models, reasoning effort,
and agents remain configurable:

```bash
python experiments/harbor_vizdoom/run_experiment.py \
  --agent codex \
  --codex-model openai/gpt-5.6-luna \
  --codex-reasoning high
```

The output directory includes the selected model, reasoning effort,
authentication mode, and actual seed. Built-in web search is disabled for
Codex and web search/fetch are disabled for Claude Code.

The launcher also applies the scoped `harbor_no_redaction/sitecustomize.py`
monkeypatch to the Harbor subprocess. It replaces Harbor's
`Trial._scrub_jobs_dir()` with a no-op because Harbor 0.22.0 otherwise treats
short authentication-control values such as `1` as secrets and corrupts every
matching character in textual results. This leaves the installed Harbor
package unchanged. It also means job logs are not scrubbed for actual secrets;
inspect them before sharing or uploading a run.

Before starting Harbor, the launcher checks that `python:3.12-slim` is present
in Docker's local cache. If it is missing, the launcher retries the pull three
times so a transient Docker Hub timeout fails early with a useful message
instead of surfacing later as a sidecar collection error. Configure this with
`--docker-pull-retries` or bypass it with `--skip-docker-preflight`.

API-key authentication is the default. Export the applicable key before the
run:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python experiments/harbor_vizdoom/run_experiment.py --agent-auth api-key
```

To use Codex and Claude subscriptions instead, first run `codex login`, then
create and export a Claude Code long-lived OAuth token with `claude setup-token`:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=...
python experiments/harbor_vizdoom/run_experiment.py \
  --agent-auth subscription
```

Codex subscription mode uses `~/.codex/auth.json` by default. Select another
file with `--codex-auth-json /path/to/auth.json`. Authentication may also be
mixed per agent, for example:

```bash
python experiments/harbor_vizdoom/run_experiment.py \
  --agent-auth subscription \
  --claude-auth api-key
```

## Random-agent smoke tests

Use the random-agent harness to test the shared server without making model API
calls. It exercises random multi-button actions, state memory, in-memory image
encoding, automap requirements, event logging, and video finalization. By
default it runs 16 actions on every manifest task with a random seed.

The local Python environment needs `gymnasium`, `vizdoom`, and `moviepy`;
FastMCP, Harbor, Docker, and model API keys are not needed for this smoke test.
Run it with:

```bash
python experiments/harbor_vizdoom/random_agent_harness.py
```

Select tasks or make a run reproducible:

```bash
python experiments/harbor_vizdoom/random_agent_harness.py \
  --task basic \
  --task doom-e1m1 \
  --seed 124343 \
  --max-actions 32
```

Each run writes `config.json`, `summary.json`, per-task server/verifier logs,
and `result.json`, `events.jsonl`, and `recording.mp4` beneath
`random_agent_runs/agent-random/reasoning-none/seed-<seed>/`. A task passes the
smoke test when the server exits cleanly and the shared Harbor verifier accepts
all three sidecar artifacts; gameplay success is reported but is not required
from a random policy.

## Generate persistent tasks

To inspect persistent tasks or create a static Harbor config, generate them
under the default gitignored directory first:

```bash
python experiments/harbor_vizdoom/generate_tasks.py
VIZDOOM_SEED=124343 harbor run \
  -c experiments/harbor_vizdoom/generated_tasks/run.yaml
```

That direct `harbor` command does not receive the launcher's monkeypatch. To
disable result redaction for a direct run too, prepend the hook directory to
`PYTHONPATH`:

```bash
PYTHONPATH="experiments/harbor_vizdoom/harbor_no_redaction${PYTHONPATH:+:$PYTHONPATH}" \
VIZDOOM_SEED=124343 harbor run \
  -c experiments/harbor_vizdoom/generated_tasks/run.yaml
```

Generate a subset or choose another output directory:

```bash
python experiments/harbor_vizdoom/generate_tasks.py \
  --task doom-e1m1 \
  --output-dir /tmp/vizdoom-tasks
```

To add another task, append one object to `environments.json`. A commercial
DOOM level should use `family: "doom"`, `doom.md.template`, its registered
Gymnasium environment ID, map metadata, and the appropriate bundled IWAD file.

ViZDoom's source documentation is in
[`docs/environments/default.md`](../../docs/environments/default.md) and
[`docs/environments/original_doom_levels.md`](../../docs/environments/original_doom_levels.md).
