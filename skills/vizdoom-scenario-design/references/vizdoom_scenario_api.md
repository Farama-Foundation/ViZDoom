# How a scenario talks to ViZDoom

Sources: `docs/environments/creating_custom.md`, `docs/api/configuration_files.md`,
`docs/api/cpp/enums.md`, `src/lib/ViZDoomGame.cpp` (`updateReward`),
`src/vizdoom/src/viz_game.cpp`, `src/vizdoom/src/g_level.cpp`, `src/lib_python/CMakeLists.txt`,
`pettingzoo_wrapper/__init__.py`. Engine: ViZDoom 1.3.2, ZDoom 2.8.1+ fork.

## A scenario is a `.wad` plus a `.cfg`

- The WAD holds one map (usually `MAP01`) in **UDMF** (`TEXTMAP` lump) or Hexen format,
  plus a `BEHAVIOR` lump with the compiled ACS (and optionally `SCRIPTS` with the source,
  `DECORATE`, `SNDINFO` and sound lumps). Doom-format maps cannot run scripts.
- No node builder is needed: the engine builds nodes when the map loads.
- The IWAD (textures, sprites, monsters, sounds) is `freedoom2.wad`, which ships in the
  `vizdoom` package next to `vizdoom.pk3` (engine resources, DECORATE actors). Find them
  with `python -c "import vizdoom, os; print(os.path.dirname(vizdoom.__file__))"`.
  `vizdoom.scenarios_path` is the package copy of `scenarios/`.

## Reward: ACS `global int 0`

```c
global int 0:reward;          // fixed point, cumulative
script 1 (void) { reward += 1.0; }   // +1.0, NOT +1
```

`DoomGame::updateReward` converts `global 0` from fixed point and returns the *change since
the previous step* as the map reward, then adds `living_reward`, `death_penalty` and the
other cfg rewards. `1` in ACS is the integer 1 = 1/65536 in fixed point; always write `1.0`.

## State exported to Python: `USER1` - `USER60`

`global int 1` - `global int 60` are copied every tic to the game variables `USER1` - `USER60`
(`viz_game.cpp`). They are raw integers; if a script stores a fixed-point value, convert it
with `vizdoom.doom_fixed_to_float`. List them in `available_game_variables` to receive them.

Treat these as the scenario's **contract** with the Python side (termination, outcome,
diagnostics), write it down in the spec (`templates/scenario_spec.md`), and:

- reset every exported global in an `OPEN` script. A single-player `new_episode()` starts a
  new game (`G_InitNew` clears all ACS variables); in a **netgame** it is a `changemap`
  (`src/lib/ViZDoomController.cpp`), which does not re-seed the RNG. A probe through the
  PettingZoo wrapper (2 players, coop and deathmatch) saw globals back at 0 after every reset,
  but the explicit reset keeps the contract independent of how the map is restarted;
- do not rely on `RESPAWN`/`ENTER` scripts to reset episode state (they also run on respawn);
- in netgames, ticcmds reach the ACS a few tics after `make_action()`, so sample the
  exports a few tics after an action before asserting on them.

## Config keys that matter for scenario design

Full list: `docs/api/configuration_files.md`. Keys are case-insensitive, underscores are
ignored, relative paths are relative to the `.cfg`, and `+=` appends to list keys.

| Key | Use |
|---|---|
| `doom_scenario_path` | the WAD (relative to the cfg) |
| `doom_map` | map name, default `map01` |
| `doom_game_path` | IWAD; leave unset to use the shipped `freedoom2.wad` |
| `game_args` | engine arguments and cvars, e.g. `-deathmatch +sv_norespawn 1 +sv_noautoaim 1` |
| `episode_timeout` | truncation in tics (35 tics = 1 s) |
| `episode_start_time` | tics skipped at the start (e.g. weapon raise) |
| `living_reward`, `death_penalty`, `kill_reward`, ... | engine-side reward terms added to the ACS reward |
| `available_buttons` | action space |
| `available_game_variables` | include `USERn` exports, `HEALTH`, `DEAD`, `PLAYER_NUMBER`, ... |
| `audio_buffer_enabled`, `audio_buffer_size`, `sound_enabled` | audio observations (needs OpenAL Soft on macOS) |
| `notifications_buffer_enabled` | text observations (chat, `HUDMSG_LOG` messages) |
| `mode` | `PLAYER` (synchronous) for training and tests |

Example: `scenarios/simple_tag_audio.cfg` (6 movement buttons, `USER50`-`USER52`
contract, audio buffer of 4 tics, `-deathmatch +sv_norespawn 1`).

## Multiplayer (netgame) scenarios

- Up to 16 players. One instance hosts (`-host N -port P`), the others join
  (`-join 127.0.0.1:P`); names and colours via `+name Player0 +color "ff 00 00"`.
  `-deathmatch` uses deathmatch starts (editor number 11) and frag rules; without it the game
  is cooperative and uses player starts 1-4 (more in `actor_classes.md`).
- Bind roles to **player names**, not network slots, if join order can vary. Simple Tag
  compares the activator's name in its player-setup script,
  `StrCmp(StrParam(n:0), StrParam(s:"Player", d:role)) == 0`, and its probe joins bots in
  shuffled orders to verify that roles follow names.
- In this fork, `vizdoom.pettingzoo_wrapper.make(config_file=..., num_agents=N)` runs every
  agent as its own networked instance and exposes a PettingZoo parallel environment
  (upstream ViZDoom does not have this wrapper).
- Respawn: `+sv_norespawn 1` prevents respawning (it overrides `sv_forcerespawn`).

## Package copy: run `make -j` after editing a WAD or cfg

`src/lib_python/CMakeLists.txt` copies `scenarios/` into `bin/pythonX/vizdoom/scenarios/`,
and Python code that loads scenarios by name (`vizdoom.scenarios_path`, the PettingZoo
`make(scenario=...)`) reads that copy. After rebuilding a WAD in `scenarios/`, run `make -j`
(or pass the source path explicitly) or you will test the old file.
