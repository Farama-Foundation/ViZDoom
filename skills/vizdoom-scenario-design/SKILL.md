---
name: vizdoom-scenario-design
description: Design, build and test ViZDoom scenarios (Doom map + ACS scripts + DECORATE + .cfg) for single- and multi-agent RL, pinned to ViZDoom's ZDoom 2.8.1 engine. Use when asked to create, modify, debug or test a ViZDoom/ZDoom WAD, ACS script, DECORATE actor or scenario config, including DoomGen or omgifol map generation and PettingZoo multiplayer scenarios.
---

# ViZDoom scenario design

A ViZDoom scenario is a WAD (map geometry, things, compiled ACS, optional DECORATE and
sounds) plus a `.cfg` (actions, observations, rewards, engine arguments). This skill takes
an agent from a written spec to a scenario that compiles, loads, satisfies its contract
with the Python side, and is ready for a human playtest.

**The engine is ZDoom 2.8.1, not GZDoom.** The ZDoom wiki documents current GZDoom, and the
`acc` compiler accepts functions that ViZDoom never runs; most such mistakes fail silently.
The reference pack in `references/` is generated from the engine sources and is the
authority on what exists.

## Ground rules

1. Use an ACS function only if it is in `references/acs_functions.md`, or a line special
   only if `references/line_specials.md` says `Works = yes`. A clean `acc` compile is not
   evidence (see `references/acs_unsupported.md`).
2. Use DECORATE parents and thing classes only from `references/actor_classes.md`.
3. Consult the wiki through `references/zdoom_wiki_index.md`, and only for semantics.
4. Do not claim a scenario works until it has passed the probe; say what is left for the
   human playtest.

## Setup check

```sh
python -c "import vizdoom, os; print(vizdoom.__version__, os.path.dirname(vizdoom.__file__))"
ls DoomGen2/acc/acc DoomGen2/acc/zcommon.acs     # ACC 1.60 shipped with COMRAD's DoomGen
python -c "import omg; print('omgifol ok')"    # pip install omgifol
```

`freedoom2.wad` and `vizdoom.pk3` ship in the `vizdoom` package directory; nothing else
needs downloading. Tool sources and install notes: `references/tooling.md`.
In the commands below, `$SKILL` is the path of this skill folder.

If DoomGen is not installed, get it at `https://github.com/TTomilin/COMRAD/tree/main/DoomGen2`.

## Workflow

```
spec -> geometry -> mechanisms (ACS) -> compile -> probe -> capture -> human playtest
          ^              ^                 |          |         |
          +--------------+-----------------+----------+---------+  (fix and repeat)
```

### 1. Specify

Copy `templates/scenario_spec.md` and fill it in: roles and how they are bound (player
names, not network slots), the `USER` contract (which ACS globals 1-60 the Python side
reads, with their values before and after each event), reward, termination, truncation,
per-episode randomisation, multiplayer mode. The contract is what the probe tests.

### 2. Build the geometry (pick a route)

- **A. Generate with DoomGen** (new layouts, seedable batches): copy the closest generator in `DoomGen2/examples/benchmark/` (e.g. `simple_tag.py`) and edit it.
- **B. Patch an existing WAD** (geometry is nearly right): edit lumps and the UDMF `TEXTMAP` directly, as the repo's `scripts/wad/build_multi_duel_hide_and_seek_wad.py` does, or use omgifol (`omg.WAD`, `omg.UMapEditor`).
- **C. Hand edit in SLADE / Ultimate Doom Builder** (UDMF, `ZDoom` port, `freedoom2.wad` + `vizdoom.pk3` resources): for a human, not the agent. Use it to inspect generated maps.

### 3. Write the mechanisms in ACS

- `#include "zcommon.acs"`; reward is `global int 0` in **fixed point** (`reward += 1.0;`).
- Export state through `global int 1..60` (seen as `USER1..USER60`), and reset all of them
  in an `OPEN` script so the contract does not depend on how the map is restarted (netgame
  episodes are map changes, and the RNG is not re-seeded).
- Per-player setup in `ENTER`/`RESPAWN` (they also run on respawn, so do not reset episode
  state there). Randomise per episode in `OPEN` with `Random()`.
- Latch outcomes in a `USER` flag and let the Python wrapper terminate and reward on it.
- Details and cfg keys: `references/vizdoom_scenario_api.md`.

### 4. Compile and package

```sh
acc -i DoomGen2/acc my_scenario.acs my_scenario.o      # DoomGen's ACSBuilder.compile() does this
make -j                                               # copy scenarios/ into the package
```

### 5. Probe in the engine (agent)

```sh
python $SKILL/scripts/validate_scenario.py --cfg scenarios/my_scenario.cfg --players 4 --episodes 3 --expect USER50=0 --expect USER51=0
```

It loads the scenario with N networked players (PettingZoo wrapper for N > 1), fails on
engine/actor-load errors, checks the `--expect` values after every reset, and checks that
episodes end within the timeout. Turn it into a regression test with
`templates/test_scenario_template.py`.

### 6. Capture and inspect (agent, multimodal)

```sh
python $SKILL/scripts/capture_multiagent_scenario.py scenarios/my_scenario.cfg --agents 4 --tics 40 --out /tmp/my_scenario_snapshot
```

Writes `summary.json` (poses, notifications, visible objects, audio peaks), `player<i>.png`,
`player<i>_audio.wav` and `topdown.png` (map lines and player poses from the WAD). Look at
them: spawns, visibility, colours, HUD text, whether sound is non-silent and directional.
On macOS the script restarts itself with OpenAL Soft on `DYLD_LIBRARY_PATH`; without it the
audio buffers are silent.

### 7. Human playtest

```sh
$SKILL/scripts/play_netgame.sh --players 4 --mode deathmatch \
    --names Player0,Player1,Player2,Player3 \
    --colors "ff 00 00,ff 00 00,ff 00 00,00 ff 00" --audio-player 0 \
    scenarios/my_scenario.wad          # --dry-run prints the commands only
```

Only a person can judge whether the scenario plays as intended (feel, fairness, audibility).
Report what was verified automatically and what still needs this playtest.

## Done checklist

- [ ] spec written, contract table complete
- [ ] every ACS call and DECORATE parent checked against `references/`
- [ ] `acc` compiles, `make -j` run
- [ ] `validate_scenario.py` passes for the intended player count
- [ ] snapshot inspected
- [ ] playtest requested, with what to look for

## Files

| Path | Purpose |
|---|---|
| `pitfalls.md` | 23 known ZDoom 2.8.1 / ViZDoom traps with symptoms and fixes |
| `references/` | version-pinned ACS, line special and actor tables; scenario API; tooling; wiki index |
| `templates/scenario_spec.md` | spec and `USER` contract template |
| `templates/test_scenario_template.py` | pytest wrapper around the probe |
| `scripts/validate_scenario.py` | engine load + contract probe |
| `scripts/capture_multiagent_scenario.py` | N-player snapshot (frames, map, audio, text) |
| `scripts/play_netgame.sh` | N-player human playtest launcher |
| `scripts/build_probe_wads.py` | builds probe WADs that reproduce the engine pitfalls and checks the validator against them |
| `scripts/build_reference_pack.py` | regenerates `references/` from the engine sources |
