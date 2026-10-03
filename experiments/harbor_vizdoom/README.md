# ViZDoom Harbor evaluation

This experiment generates self-contained Harbor tasks from a single template
and a manifest. The included task definitions are:

| Slug | Gymnasium environment | Objective |
| --- | --- | --- |
| `basic` | `VizdoomBasic-MultiBinary-v1` | Kill the monster |
| `my-way-home` | `VizdoomMyWayHome-MultiBinary-v1` | Reach the armor vest |
| `defend-the-center` | `VizdoomDefendCenter-MultiBinary-v1` | Maximize combat return until death/timeout |
| `health-gathering` | `VizdoomHealthGathering-MultiBinary-v1` | Maximize survival time until death/timeout |
| `doom-e1m1` | `VizdoomDoomE1M1-S1-v0` | Complete Hangar |
| `doom-e1m2` | `VizdoomDoomE1M2-S1-v0` | Complete Nuclear Plant |
| `doom2-map01` | `VizdoomDoom2MAP01-S1-v0` | Complete Entryway |
| `doom2-map02` | `VizdoomDoom2MAP02-S1-v0` | Complete Underhalls |

All environments use a Gymnasium `MultiBinary` action space and frame skip 4.
Commercial DOOM maps run at skill level 1 and provide both the first-person
screen and automap. The included Basic and My Way Home scenarios return the
first-person screen only. The two additional standard scenarios also return
the first-person screen only.

## Source structure

```text
harbor_vizdoom/
├── environments.json                 # Task/map definitions
├── generate_tasks.py                 # Task generator
├── harbor_no_redaction/
│   ├── sitecustomize.py              # Scoped Harbor hooks
│   └── episode_continuation.py       # Resume Codex until game ends
├── random_agent_harness.py           # Local random-policy smoke tests
├── run_experiment.py                 # Generate temporarily and run Harbor
├── run_pilot.py                      # Five paired seeds, selectable GPT-5.6/6 models
├── export_paper_results.py           # Validated table and supporting per-seed data
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
  named buttons for 1-2 repetitions and returns the new state. Each repetition
  advances up to four game tics. The two short
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
finalizes it even when the agent stops before completing the episode. Such an
early stop is labeled separately and is not a natural episode ending. The
Codex launcher checks the live sidecar after each CLI turn and resumes the
same conversation if the episode is still running. It never chooses buttons
for the model. Provider failures and any continuation-limit interruption
remain invalid attempts with their artifacts retained. The
verifier scores goal completion for Basic and My Way Home, native return for
Defend the Center, and simulated survival seconds for Health Gathering. It
does not treat positive intermediate reward as a completed episode.

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
random unsigned 32-bit seed, and runs all eight tasks:

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

The four-scenario Luna/Terra/Sol pilot uses five paired random seeds and Codex
subscription authentication by default:

```bash
python experiments/harbor_vizdoom/run_pilot.py
```

By default it runs 60 trials in fifteen Harbor jobs and may take several hours.
Use `--models` to select one or more models. For example, to extend the original
Luna/Terra pilot with Sol on the same seeds and the same in-container CLI version:

```bash
python experiments/harbor_vizdoom/run_pilot.py \
  --models openai/gpt-5.6-sol --codex-version 0.155.1 \
  --seeds 619549730 2395780781 74702573 450023666 4170563159 \
  --parallel-jobs 2
```

This launches 20 Sol trials in a new directory and preserves the earlier runs.
`--codex-version` pins the trial CLI; the host CLI version can differ. The pilot
records `protocol.json`, incremental `runs.jsonl`, Harbor job artifacts, and
per-job logs under `pilot_runs/<timestamp>/`. Each trial stores the sidecar
video and events plus the Codex transcript and trajectory. To resume a pilot,
pass `--resume-dir` with its exact run directory. Set `--seeds` to five distinct
unsigned 32-bit numbers to prespecify starting states. Two task trials run
concurrently by default; `--concurrent-trials 1` uses less Docker memory.
For a long batch, `--parallel-jobs 2` overlaps two independent Harbor jobs
without changing per-trial control settings. Run `summarize_pilot.py <run-dir>`
after a job finishes; it reports natural endings and interruptions separately.
For a read-only snapshot while a multi-task job is still running, use
`summarize_pilot.py <run-dir> --include-completed-trials`. This writes
`live_summary.json`, including finished, validated individual trials and their
artifact paths without modifying `runs.jsonl` or the supervisor's `summary.json`.
The Codex web-search tool is disabled. Report incomplete episodes separately;
do not silently replace or omit them when analyzing the pilot.

The six selectable configurations are GPT-5.6 Luna/Terra/Sol, GPT-6 Luna/Sol,
and GPT-6.1 Sol. The default remains the original three GPT-5.6 models.
Subscription probes on October 1 found that GPT-6 Luna/Sol work with CLI
0.155.1, while GPT-6.1 Sol works with 0.159.2 but is rejected by 0.155.1.
The expanded comparison pins **0.159.2** rather than silently changing auth or
substituting a model. Model access is account- and release-dependent.

Use `--detach` for a persistent runner and supervisor whose stdout/stderr go to
`runner.log` and `supervisor.log`, not the launching client's pipe. Both process
IDs and commands are saved in `launcher.json`. `--prepare-only` saves and checks
the protocol without starting trials. New pilots execute the archived source
tree, verify its hashes, and keep commercial IWADs as checked local links rather
than archiving them or copying credentials.

`--reuse-completed-from <run-dir>` can carry verified same-seed natural outcomes
into an expanded design. Controls and gameplay settings must match. Natural
deaths and timeouts are reused too, never replayed to improve their score.
Their original source/CLI provenance remains in `protocol.json`; only missing
task cells are launched. Interrupted originals remain in their original audit
directory and are not counted as gameplay outcomes.

Select manifest tasks with `--tasks`. For fifteen DOOM 1 E1M1 (Hangar) trials,
five newly randomized paired seeds for Luna, Terra, and Sol:

```bash
python experiments/harbor_vizdoom/run_pilot.py \
  --tasks doom-e1m1 --codex-version 0.155.1 \
  --parallel-jobs 3 --concurrent-trials 1
```

E1M1 runs at skill 1 for at most **600 seconds of in-game time** (21,000 tics),
or until level completion/death, whichever happens first. The shared server
sets the engine's episode timeout; the 5,250 four-tic action budget is large
enough to reach it even with `repeat=1`. Frame skip 4 and maximum repeat 2 are
unchanged. Inference does not advance game time, so a run may take hours of
wall-clock time; the 24-hour agent timeout is an infrastructure safeguard,
not the gameplay limit. Such interruptions are not valid completed episodes.
Screen, automap, event logs, video, and agent transcript are retained.
E1M2 (Nuclear Plant) and Doom II MAP02 (Underhalls) have **1,200-second** game
limits, 10,500-step budgets, and 48-hour outer agent guards. Doom II MAP01
(Entryway) has the same 600-second/5,250-step/24-hour limits as Hangar. These
are independent skill-1 levels, not a persistent campaign. Standard scenarios
retain their native engine limits. The protocol records the selected task definitions.

For a six-model, four-map design with five paired seeds (120 outcome cells):

```bash
python experiments/harbor_vizdoom/run_pilot.py \
  --models openai/gpt-5.6-luna openai/gpt-5.6-terra openai/gpt-5.6-sol \
    openai/gpt-6-luna openai/gpt-6-sol openai/gpt-6.1-sol \
  --tasks doom-e1m1 doom-e1m2 doom2-map01 doom2-map02 \
  --codex-version 0.159.2 --parallel-jobs 1 --concurrent-trials 1 --detach
```

Add the original five Hangar `--seeds` and `--reuse-completed-from` its run
directory to preserve already completed outcomes when extending that batch.
Long level trials can take many wall-clock hours. Keep total concurrency across
all active pilots within the available Docker memory and subscription limits.

For long runs, start a supervisor in another terminal using the printed run
directory (it does not launch duplicate original trials):

```bash
python experiments/harbor_vizdoom/supervise_pilot.py <run-dir> --manage-memory
```

It refreshes `summary.json` as trials finish, then runs the existing restricted
infrastructure-recovery policy and checks every selected video with FFmpeg.
Completion produces `artifact_validation.json`. Only one supervisor can hold
the run's lock. Progress, errors, and scheduling changes are appended to
`supervision.jsonl`.

The shared server now uses a streaming subclass of Gymnasium `RecordVideo`:
RGB frames are sent directly to a bounded FFmpeg pipe and no frame PNGs or
whole-episode frame list are kept. First-person and automap views remain
side-by-side in `recording.mp4`. Exact-container tests on all four maps and a
10,501-frame recording verified finalization and bounded memory.

Earlier runs used Gymnasium's whole-episode RAM buffer. With `--manage-memory`, the
supervisor pauses this run's agent containers when Docker VM available memory
falls below 2 GiB, keeping the furthest-progressed episode running. It resumes
them once available memory reaches 3.5 GiB. Game servers are not paused or reset;
game time advances only on actions. These logged pauses do affect wall-clock
latency and the outer agent timeout, so paused runs are unsuitable for unadjusted
wall-clock comparisons. Omit the flag to disable container pausing; summary
updates, restricted recovery, and final artifact checks still run.

If a trial is interrupted by a verified provider `ApiOverloadedError`, or if
an earlier unguarded trial ended while the game was still running, keep the
original attempt and retry only that task on the same seed, for example:

```bash
python experiments/harbor_vizdoom/retry_infrastructure.py \
  experiments/harbor_vizdoom/pilot_runs/20260921T224407Z \
  --model openai/gpt-5.6-terra \
  --seed 2395780781 \
  --task health-gathering
```

Retry attempts are appended to `retry_attempts.jsonl`; the summarizer uses a
validated natural-ending retry for that seed only when its original attempt is
invalid, and still reports the original interruption. `EarlyAgentExit`
retries are labeled separately from provider-capacity retries. Never score a
partial episode as a natural death, timeout, or success.
After all main Harbor jobs have finished, `recover_pilot.py <run-dir>`
checks for unlaunched cells, retries eligible interruptions sequentially, and
exits successfully only when every requested model/task/seed cell has a validated natural
ending. It preserves the original attempts and does not retry natural gameplay
failures.

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
The same scoped hook resumes Codex after a completed CLI turn whenever the
sidecar still reports `running`; resumed turns append to `agent/codex.txt`.

Before starting Harbor, the launcher checks that `python:3.12-slim` is present
in Docker's local cache. It first checks Docker daemon/socket access, so a
permission error is not mistaken for a missing image. If the image is missing,
the launcher retries the pull three
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
