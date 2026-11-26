---
name: vizdoom-scenario-design
description: Design, build and test ViZDoom scenarios (Doom map + ACS scripts + DECORATE + .cfg) for single- and multi-agent RL, pinned to ViZDoom's ZDoom 2.8.1 engine. Use when asked to create, modify, debug or test a ViZDoom/ZDoom WAD, ACS script, DECORATE actor or scenario config, including DoomGen or omgifol map generation and multiplayer scenarios.
---

# ViZDoom scenario design

A ViZDoom scenario is a WAD (map geometry, things, compiled ACS, optional DECORATE and
sounds) plus a `.cfg` (actions, observations, rewards, engine arguments). This skill takes
an agent from a written spec to a scenario that compiles, loads, satisfies its contract
with the Python side, and is ready for a human playtest.

**The engine is ZDoom 2.8.1, not GZDoom.** The ZDoom wiki documents current GZDoom, and the
`acc` compiler accepts functions that ViZDoom never runs; most such mistakes fail silently.
The reference pack in `references/` is generated from the engine sources and is the
authority on what exists; `scripts/validate_scenario.py` enforces it automatically.

## Ground rules

1. Pick ACS functions from `references/acs_functions.md`, line specials with `Works = yes`
   in `references/line_specials.md`, and DECORATE parents from
   `references/actor_classes.md`. A clean `acc` compile is not evidence.
2. Use the wiki (`references/zdoom_wiki_index.md`) only for semantics.
3. Do not claim a scenario works until `validate_scenario.py` passes; state what remains
   for the human playtest.

## Setup

`$SKILL` is the path of this skill folder. Everything except DoomGen2's geometry packages
(and optional snapshot plotting packages) ships with the skill or `vizdoom`:

```sh
pip install omgifol shapely scipy networkx                 # DoomGen2 dependencies
pip install pillow matplotlib                              # optional: capture PNGs and top-down plots
python -c "import vizdoom; print(vizdoom.__version__)"
export PYTHONPATH=$SKILL/DoomGen2/src                      # or: pip install -e $SKILL/DoomGen2
```

`$SKILL/DoomGen2` is a vendored map generator with a Python ACS builder, ACC 1.60 in `acc/`
and ZDBSP in `zdbsp/`, both found automatically. Bundled `acc` and `zdbsp` are macOS/Windows
binaries; on Linux build `acc` from <https://github.com/rheit/acc> and pass `--acc` (ZDBSP
is optional — DoomGen2 skips it automatically if unavailable and the engine builds nodes at
load time). Details: `references/tooling.md`.

## Workflow

```
spec -> geometry -> mechanisms (ACS) -> compile -> probe -> capture -> human playtest
          ^              ^                 |          |         |
          +--------------+-----------------+----------+---------+  (fix and repeat)
```

### 1. Specify

Copy `templates/scenario_spec.md` and fill it in: roles and how they are bound (player
names, not network slots), the `USER` contract (ACS globals 1-60 read by Python, before and
after each event), reward, termination, truncation, per-episode randomisation, multiplayer
mode. The contract is what the probe tests.

### 2. Build the geometry (pick a route)

- **A. Generate with DoomGen2** (new layouts, seedable batches): start from the closest
  example in `$SKILL/DoomGen2/examples/` (indexed in `DoomGen2/examples/README.md`:
  `geometry_and_placement.py`, `key_maze.py`, `coop_relay.py`, `simple_tag.py`,
  `deathmatch_arena.py`). Every script takes `--output <path.wad>` (defaulting to
  `DoomGen2/examples/output/`, git-ignored); write the `.cfg` yourself.
- **B. Patch an existing WAD** (geometry is nearly right): edit lumps and maps via
  `omgifol` (`omg.WAD`, `omg.UMapEditor`, `omg.MapEditor`) or direct WAD/UDMF `TEXTMAP`
  patching, then recompile `BEHAVIOR` with `acc`.
- **C. Hand edit in SLADE / Ultimate Doom Builder**: for humans inspecting or authoring
  UDMF maps.

### 3. Write the mechanisms in ACS

- `#include "zcommon.acs"`; reward is `global int 0` in **fixed point** (`reward += 1.0;`,
  never `+= 1` or `++`).
- Export state through `global int 1..60` (seen as `USER1..USER60`), and **reset every one
  of them in an `OPEN` script**. Single-player `new_episode()` runs `G_InitNew` (clears
  globals), but a netgame `new_episode()` runs `changemap` (`P_ClearACSVars(false)`), so
  `global int 0..60` persist across netgame episodes unless `OPEN` resets them, and
  `Random()` continues its stream.
- Put per-player setup in `ENTER`/`RESPAWN` (do not reset episode state there). Randomise
  per episode in `OPEN` with `Random()`.
- Latch outcomes in a `USER` flag and let Python terminate and reward on it.
- Details and cfg keys: `references/vizdoom_scenario_api.md`. Known traps: `pitfalls.md`.

### 4. Compile and package

```sh
$SKILL/DoomGen2/acc/acc -i $SKILL/DoomGen2/acc my_scenario.acs my_scenario.o  # DoomGen2's ACSBuilder.compile() does this
make -j                                                                      # copy scenarios/ into the package
```

### 5. Probe in the engine (agent)

```sh
python $SKILL/scripts/validate_scenario.py --cfg scenarios/my_scenario.cfg --players 4 \
    --episodes 3 --expect USER50=0 --expect USER51=0 --expect-idle USER50=0
```

Three stages, stopping on first failure:
1. **Static lint**: map format, spawns, `ZSCRIPT`, uncompiled `SCRIPTS`, UDMF specials > 255,
   DECORATE parents/`replaces`, unsupported ACS calls, integer reward literals, swapped
   `PlaySound` args, non-literal `SpawnSpot` classes, stale/missing package copy.
2. **Engine load**: isolated single-player load checking engine console errors and `.cfg`
   key/value warnings.
3. **Contract probe**: native ViZDoom 1-16 player run checking `--expect` after every reset
   (including real netgame `changemap` resets), `--expect-idle`, idle survival, and optional
   `--check-timeout`.

Turn the probe into a pytest regression test with `templates/test_scenario_template.py`.

### 6. Capture and inspect (agent, multimodal)

```sh
python $SKILL/scripts/capture_multiagent_scenario.py scenarios/my_scenario.cfg \
    --agents 4 --tics 40 --out /tmp/my_scenario_snapshot
```

Writes `overview.png` (top-down map + every player's POV in one figure), `summary.json`
(poses, health, `dead`/`dead_steps`, notifications, visible objects, audio peaks),
`player<i>.png`, `player<i>_audio.wav` (when audio is enabled in the `.cfg`) and
`topdown.png`. **Inspect `overview.png` and `summary.json` first** to verify spawns (no
`DEAD` players), geometry, visibility, colours, HUD text, and non-silent directional audio.
On macOS the script re-execs itself with OpenAL Soft on `DYLD_LIBRARY_PATH`.

### 7. Human playtest

```sh
$SKILL/scripts/play_netgame.sh --players 4 --mode deathmatch \
    --names Player0,Player1,Player2,Player3 \
    --colors "ff 00 00,ff 00 00,ff 00 00,00 ff 00" --audio-player 0 \
    scenarios/my_scenario.wad          # --dry-run prints the commands only
```

Only a person can judge feel, fairness and audibility. Report what passed automatically and
what still needs this playtest.

## Done checklist

- [ ] spec written, `USER` contract table complete
- [ ] `acc` compiles cleanly, `make -j` run (if under `scenarios/`)
- [ ] `validate_scenario.py` passes for the intended player count
- [ ] `overview.png` and `summary.json` inspected (all players alive, geometry and POVs match spec)
- [ ] playtest requested, with what to look for

## Files

| Path | Purpose |
|---|---|
| `pitfalls.md` | 23 ZDoom 2.8.1 / ViZDoom traps, fixes, and which automated check enforces each |
| `references/` | engine-generated ACS, line special and actor tables; scenario API; tooling; wiki index |
| `templates/` | spec and `USER` contract template; pytest wrapper around `validate_scenario.py` |
| `scripts/validate_scenario.py` | static lint + engine load + native 1-16 player contract probe |
| `scripts/capture_multiagent_scenario.py` | native 1-16 player snapshot (`overview.png`, frames, top-down map, audio, notifications) |
| `scripts/play_netgame.sh` | N-player human playtest launcher |
| `scripts/build_probe_wads.py` | probe WADs reproducing engine pitfalls; `--check` tests the validator against them |
| `scripts/build_reference_pack.py` | regenerates `references/` deterministically from engine sources |
| `DoomGen2/` | vendored map generator, ACS builder, `acc`, `zdbsp`, and 5 worked examples in `examples/` |

The skill's own regression tests are `tests/test_scenario_skill.py` (automated) and
`tests/manual_test_scenario_skill.py` (full netgame probe check) in the repository.
