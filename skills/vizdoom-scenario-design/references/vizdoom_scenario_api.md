# How a scenario talks to ViZDoom

Sources: `docs/environments/creating_custom.md`, `docs/api/configuration_files.md`,
`docs/api/cpp/enums.md`, `src/lib/ViZDoomGame.cpp` (`updateReward`),
`src/lib/ViZDoomController.cpp` (`setMap`), `src/vizdoom/src/viz_game.cpp`,
`src/vizdoom/src/g_level.cpp`, `src/lib_python/CMakeLists.txt`.
Engine: ViZDoom 1.3.3, ZDoom 2.8.1+ fork.

## A scenario is a `.wad` plus a `.cfg`

- The WAD holds one map (usually `MAP01`) in **UDMF** (`TEXTMAP` lump) or Hexen format,
  plus a `BEHAVIOR` lump with the compiled ACS (and optionally `SCRIPTS` with the source,
  `DECORATE`, `MAPINFO`, `SNDINFO` and sound lumps). Doom-format maps cannot run scripts.
- No node builder is needed: the engine builds nodes when the map loads.
- The IWAD (textures, sprites, monsters, sounds) is `freedoom2.wad`, which ships in the
  `vizdoom` package next to `vizdoom.pk3` (engine resources, DECORATE actors). Find them
  with `python -c "import vizdoom, os; print(os.path.dirname(vizdoom.__file__))"`.
  `vizdoom.scenarios_path` is the package copy of `scenarios/`.

## Reward: ACS `global int 0`

```c
global int 0:reward;                 // fixed point, cumulative
script 1 (void) { reward += 1.0; }   // +1.0, NOT +1 or reward++
```

`DoomGame::updateReward` converts `global 0` from fixed point and returns the *change since
the previous step* as the map reward, then adds `living_reward`, `death_penalty` and the
other cfg rewards. `1` in ACS is the integer 1 = 1/65536 in fixed point; always write `1.0`.

## State exported to Python: `USER1` - `USER60`

`global int 1` - `global int 60` are copied every tic to `USER1` - `USER60` (`viz_game.cpp`).
They are raw integers; convert fixed-point values with `vizdoom.doom_fixed_to_float`. List
them in `available_game_variables` to receive them.

Treat these as the scenario's **contract** with Python (termination, outcome, diagnostics),
write it down in the spec (`templates/scenario_spec.md`), and:

- **reset every exported global (`global int 0..60`) in an `OPEN` script.** Single-player
  `new_episode()` runs `map` -> `G_InitNew` -> `P_ClearACSVars(true)` (clears globals), but
  a **netgame** `new_episode()` runs `changemap` -> `G_DoCompleted` -> `P_ClearACSVars(false)`
  (`src/vizdoom/src/g_level.cpp`), which leaves `ACS_GlobalVars` intact across episodes and
  does not re-seed the RNG;
- do not rely on `RESPAWN`/`ENTER` scripts to reset episode state (they also run on respawn);
- in netgames, ticcmds reach ACS a few tics after `make_action()`, so sample exports a few
  tics after an action before asserting on them.

## Config keys that matter for scenario design

Full list: `docs/api/configuration_files.md`. Keys are case-insensitive, underscores are
ignored, relative paths are relative to the `.cfg`, and `+=` appends to list/string keys.
Enum values must use enum names (e.g. `audio_sampling_rate = SR_22050`, not `22050`);
`validate_scenario.py` fails on ViZDoom's `WARNING! Loading config from:` lines.

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
| `audio_buffer_enabled`, `audio_buffer_size`, `audio_sampling_rate`, `sound_enabled` | audio observations (needs OpenAL Soft on macOS; see `scenarios/basic_audio.cfg`) |
| `notifications_buffer_enabled` | text observations: chat and `HUDMSG_LOG` messages (see `scenarios/basic_notifications.cfg`) |
| `mode` | `PLAYER` (synchronous) for training and tests |

## Multiplayer (netgame) scenarios

- Up to 16 players. One instance hosts (`-host N -port P`), the others join
  (`-join 127.0.0.1:P`); names and colours via `+name Player0 +color "ff 00 00"`.
  `-deathmatch` uses deathmatch starts (editor number 11) and frag rules; without it the
  game is cooperative and uses deterministic player starts 1-4 (more in `actor_classes.md`).
  In ZDoom 2.8.1 `-deathmatch` level loads pick starts randomly with replacement (even with
  `+sv_spawnfarthest 1`), so use a `DECORATE` `+NODAMAGE` player class + `ENTER` `MapSpot`
  relocation when you need telefrag-free `-deathmatch` spawns (pitfall 23).
- Bind roles to **player names**, not network slots, if join order can vary. See
  `DoomGen2/examples/simple_tag.py` (`StrCmp(StrParam(n:0), StrParam(s:"Player", d:role)) == 0`).
- Respawn: `+sv_norespawn 1` prevents respawning (it overrides `sv_forcerespawn`).

## Package copy: run `make -j` after editing a WAD or cfg

`src/lib_python/CMakeLists.txt` copies `scenarios/` into `bin/pythonX/vizdoom/scenarios/`,
and code that loads scenarios via `vizdoom.scenarios_path` reads that copy. After editing a
WAD or cfg in `scenarios/`, run `make -j` (or pass explicit paths); `validate_scenario.py`
fails when the package copy is missing or out of date.
