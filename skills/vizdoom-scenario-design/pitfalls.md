# ViZDoom (ZDoom 2.8.1) scenario pitfalls

Read this before writing ACS, DECORATE or a scenario test. Every entry was hit while
building the multi-agent scenarios of this repository, follows from the engine source, or
was reproduced with small probe WADs when this skill was written ("probe", 2026-10-26: a
DoomGen arena with marker globals, run through `scripts/validate_scenario.py`); the source
column says which. Most of these failures are **silent**: `acc` compiles, the WAD loads,
and the scenario just behaves differently.

## Engine version (ACS and actors)

| # | Pitfall | Symptom | Fix | Source |
|---|---|---|---|---|
| 1 | Calling a GZDoom/Zandronum/Eternity extension function, e.g. `SetActivatorToPlayer` | compiles; the call silently returns 0 and does nothing | use only names in `references/acs_functions.md`; here `SetActivator(tid)` | `docs/multi-agent-scenarios.md` (`coop_relay`); probe: the script continued past the call |
| 2 | Calling a p-code built-in the engine lacks (Skulltag team/key functions, `IsNetworkGame`, `ConsoleCommand` with constant args, ...) | compiles; the whole script stops at the call, and only the engine console says `Unknown P-Code 118 in script 1` | same as 1 | `src/vizdoom/src/p_acs.cpp` (interpreter `default:` case); probe: a marker set after `IsNetworkGame()` stayed 0 |
| 3 | Line specials that exist in the wiki but map to `LS_NOP` here, or are numbered above 255 (e.g. `Sector_SetFloorGlow`) | the special does nothing | check `Works = yes` in `references/line_specials.md` | `src/vizdoom/src/p_lnspec.cpp` |
| 4 | DECORATE parent class that does not exist in 2.8.1, e.g. `ACTOR PowerTagInvisibility : PowerupNone` | actor loading fails (`Parent type 'PowerupNone' not found in PowerTagInvisibility`), no ACS runs, the Python process exited with code 139 | check the parent in `references/actor_classes.md`; for invisibility use render style `STYLE_None` and zero alpha, as Simple Tag Audio's prey does | `DoomGen/examples/benchmark/simple_tag.md` ("Reproduced failure"); probe: same console error, exit code 139 |
| 5 | Wrong `PlaySound` argument order | the sound does not play as intended | `PlaySound(tid, sound, channel, volume, looping, attenuation)` | `DoomGen/examples/benchmark/simple_tag.md` |
| 6 | Passing a runtime `str` variable as the class name of `SpawnSpot`/`SpawnSpotFacing` | spawning is not reliable in 2.8.1 (avoid) | branch to calls with literal class names; check `ThingCount(T_NONE, tid)` before retagging with `Thing_ChangeTID` | `DoomGen/AGENTS.md` |
| 7 | Relying on `NOTELEFRAG` or `INVULNERABLE` to protect players at spawn | players die at map start when spawns overlap (the spawn-stomp path ignores `NOTELEFRAG`, invulnerability does not stop telefrag damage) | give players `NODAMAGE`/`NOPAIN` for a settling period, then clear them | `DoomGen/examples/benchmark/simple_tag.md` |
| 8 | Weapon with `+WEAPON.NOAUTOFIRE` under a held `ATTACK` | after the weapon clicks empty once, a held `ATTACK` never fires again (one rocket per round) | derive a weapon without the flag (`DuelRocketLauncher`) | `docs/multi-agent-scenarios.md` (`multi_duel_hide_and_seek`) |

## ViZDoom interface (reward, exports, episodes)

| # | Pitfall | Symptom | Fix | Source |
|---|---|---|---|---|
| 9 | Writing `reward += 1` instead of `reward += 1.0` in `global int 0` | reward is 65536 times too small (`1` = 1/65536 in fixed point) | always use a decimal point for the reward | `docs/environments/creating_custom.md` |
| 10 | Expecting a netgame episode to be replayable by seed | `new_episode()` in a netgame is a map change: `Random()` continues its stream (episodes differ, peers agree) | randomise in `OPEN` and do not rely on seeds for per-episode layouts; also reset every exported global in `OPEN` (the probe saw them cleared on reset in coop and deathmatch netgames, but the explicit reset documents the contract) | `docs/multi-agent-scenarios.md` (`coop_relay`); probe (globals) |
| 11 | Reading exports right after `make_action()` in a netgame | assertions see the previous state | ticcmds reach the ACS a few tics later; sample a few tics after the action | `docs/multi-agent-scenarios.md` (`coop_relay`) |
| 12 | Terminating or rewarding on engine events (death, frags) instead of an explicit flag | double counting or wrong winner (e.g. the prey dies *after* the catch latch; frag credit is not the catcher) | latch the outcome in a `USER` flag; terminate and reward in the wrapper on that flag only | `DoomGen/examples/benchmark/simple_tag.md` |
| 13 | Dead players respawn and reopen a finished round | a second outcome in the same episode | `+sv_norespawn 1` in `game_args` (it overrides `sv_forcerespawn`); refuse to restart a latched round in `RESPAWN` | `DoomGen/examples/benchmark/simple_tag.md` |
| 14 | Binding roles to network slots | roles swap when clients join in another order | bind roles to `+name` (`StrCmp(StrParam(n:0), ...)`); test shuffled join orders | `DoomGen/examples/benchmark/simple_tag.md` |
| 15 | Editing `scenarios/x.wad` and testing by scenario name | the old WAD runs (training and `vizdoom.scenarios_path` read the package copy) | run `make -j` after every WAD/cfg change, or pass explicit paths | `docs/multi-agent-scenarios.md`, `src/lib_python/CMakeLists.txt` |
| 16 | Frame 0 checks in multiplayer | a green `TeleportFog` is visible at the player starts for ~1 s | skip the first second in visual checks | `docs/multi-agent-scenarios.md` |
| 17 | Feeding chat/HUD notifications to a text model verbatim | `\x1c` colour escapes and console bar characters in the text | strip control characters first | `docs/multi-agent-scenarios.md` |

## Audio

| # | Pitfall | Symptom | Fix | Source |
|---|---|---|---|---|
| 18 | Engine built with OpenAL but without libsndfile | native Doom sounds play, custom WAV lumps are silently missing | build with libsndfile; check that the audio buffer is non-silent after the spawn sounds | `docs/simple_tag.md` (HPC audio) |
| 19 | macOS without OpenAL Soft on the library path | silent audio buffer | `DYLD_LIBRARY_PATH=/opt/homebrew/opt/openal-soft/lib` (`play_netgame.sh` finds it) | `docs/simple_tag.md` |
| 20 | Several playtest clients playing sound at once | overlapping spatial mixes mislead the tester | enable sound on one listener, `-nosound` on the others (`--audio-player`) | `docs/simple_tag.md` |

## Map geometry and spawns

| # | Pitfall | Symptom | Fix | Source |
|---|---|---|---|---|
| 21 | `add_area` without `connect_adjacent` | a solid, full-height wall between neighbouring areas | `connect_adjacent(a, b)` for a two-sided portal | `DoomGen/AGENTS.md` |
| 22 | `add_boundary` doors on some Voronoi layouts | unreliable doors | build explicit door sectors | `DoomGen/AGENTS.md` (`coop_puzzle.py`) |
| 23 | `-deathmatch` in `game_args` but no deathmatch starts (editor number 11) in the map | the engine crashes while loading (exit code 139) without a console message | add deathmatch starts (one per player avoids spawn telefrags); `validate_scenario.py` checks this | probe |
