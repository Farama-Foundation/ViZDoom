# ViZDoom (ZDoom 2.8.1) scenario pitfalls

Read this before writing ACS, DECORATE or a scenario test. Most failures below are
**silent**: `acc` compiles, the WAD loads, and the scenario simply behaves differently.
Where possible, `scripts/validate_scenario.py` catches the mistake automatically; the
`Enforced by` column marks which verification layer guards each pitfall.

## Engine version (ACS and actors)

| # | Pitfall | Symptom | Fix | Enforced by |
|---|---|---|---|---|
| 1 | Calling a GZDoom/Zandronum/Eternity extension function, e.g. `SetActivatorToPlayer` | compiles; call silently returns 0 and does nothing | use only names in `references/acs_functions.md` (e.g. `SetActivator(tid)`) | `validate_scenario.py` (static) |
| 2 | Calling a p-code built-in the engine lacks (`IsNetworkGame`, `ConsoleCommand`, Skulltag team/key functions) | compiles; the whole script aborts at the call with `Unknown P-Code` on the engine console | same as 1 | `validate_scenario.py` (static + load) |
| 3 | Line specials that map to `LS_NOP` or are numbered above 255 (e.g. `Sector_SetFloorGlow`) | the special does nothing | check `Works = yes` in `references/line_specials.md` | `validate_scenario.py` (static) |
| 4 | DECORATE parent or `replaces` class missing in 2.8.1, e.g. `ACTOR PowerTagInvisibility : PowerupNone` | actor loading fails (`Parent type 'PowerupNone' not found`), no ACS runs, process crashes (signal 11 / exit 139) | check `references/actor_classes.md`; for invisibility use `STYLE_None` and zero alpha (`DoomGen2/examples/simple_tag.py`) | `validate_scenario.py` (static + load) |
| 5 | Swapped `PlaySound` argument order | sound does not play as intended | `PlaySound(tid, sound, channel, volume, looping, attenuation)` | `validate_scenario.py` (static) |
| 6 | Passing a runtime `str` variable as the class name of `SpawnSpot`/`SpawnSpotFacing` | spawning is unreliable in 2.8.1 | branch to calls with literal class names; check `ThingCount(T_NONE, tid)` before retagging (`DoomGen2/examples/deathmatch_arena.py`) | `validate_scenario.py` (static) |
| 7 | Relying on `NOTELEFRAG` or `INVULNERABLE` to protect players at spawn | players die at map start when spawns overlap (spawn-stomp ignores `NOTELEFRAG` and invulnerability) | give players `NODAMAGE`/`NOPAIN` during a settling period, then clear them (`DoomGen2/examples/simple_tag.py`) | `validate_scenario.py` (contract idle check) |
| 8 | Weapon with `+WEAPON.NOAUTOFIRE` under a held `ATTACK` | after clicking empty once, a held `ATTACK` never fires again | derive a weapon in DECORATE without `WEAPON.NOAUTOFIRE` | manual / playtest |

## ViZDoom interface (reward, exports, episodes)

| # | Pitfall | Symptom | Fix | Enforced by |
|---|---|---|---|---|
| 9 | Writing `reward += 1` or `reward++` instead of `reward += 1.0` in `global int 0` | reward is 65536 times too small (`1` = 1/65536 in fixed point) | always use fixed-point literals (`1.0`, `0.5`) for `global int 0` | `validate_scenario.py` (static) |
| 10 | Assuming netgame resets clear globals or re-seed RNG | netgame `new_episode()` runs `changemap` (`P_ClearACSVars(false)` in `g_level.cpp`): `global int 0..60` persist across episodes and `Random()` continues its stream | explicitly reset every exported global (`global int 0..60`) in `OPEN`; randomise per episode in `OPEN` without expecting seed replay | `validate_scenario.py` (static note + netgame contract probe) |
| 11 | Reading exports immediately after `make_action()` in a netgame | assertions see the previous state | ticcmds reach ACS a few tics later; sample a few tics after the action | test design |
| 12 | Terminating or rewarding on engine events (death, frags) instead of an explicit flag | double counting or wrong winner (e.g. prey dies *after* catch latch; frag credit differs from catcher) | latch the outcome in a `USER` flag; terminate and reward in Python on that flag only (`DoomGen2/examples/simple_tag.py`) | spec + contract test |
| 13 | Dead players respawn and reopen a finished round | a second outcome occurs in the same episode | add `+sv_norespawn 1` in `game_args` (overrides `sv_forcerespawn`); guard `RESPAWN` against latched rounds | `validate_scenario.py` (static note) |
| 14 | Binding roles to network slots (`PlayerNumber()`) | roles swap when clients join in another order | bind roles to `+name` (`StrCmp(StrParam(n:0), ...)`) (`DoomGen2/examples/simple_tag.py`); test shuffled join orders | spec + custom probe |
| 15 | Editing `scenarios/x.wad` and testing by scenario name | the old WAD runs (`vizdoom.scenarios_path` reads the built package copy) | run `make -j` after every edit under `scenarios/`, or pass explicit paths | `validate_scenario.py` (static) |
| 16 | Visual assertions on frame 0 in multiplayer | green `TeleportFog` obscures player starts for ~1 s | skip the first second in visual checks | `capture_multiagent_scenario.py` |
| 17 | Feeding chat/HUD notifications to a text model verbatim | `\x1c` colour escapes and console bar characters pollute the text | strip control characters first | wrapper / preprocessing |

## Audio

| # | Pitfall | Symptom | Fix | Enforced by |
|---|---|---|---|---|
| 18 | Engine built with OpenAL but without libsndfile | native Doom sounds play, custom WAV lumps are silently missing | build with libsndfile; verify non-silent `audio_peak_left_right` after spawn sounds | `capture_multiagent_scenario.py` |
| 19 | macOS without OpenAL Soft on `DYLD_LIBRARY_PATH` | silent audio buffer | `DYLD_LIBRARY_PATH=/opt/homebrew/opt/openal-soft/lib` (handled automatically by the skill scripts) | skill scripts |
| 20 | Several playtest clients playing sound at once | overlapping spatial mixes mislead the tester | enable sound on one listener, `-nosound` on the others (`play_netgame.sh --audio-player 0`) | `play_netgame.sh` default |

## Map geometry and spawns

| # | Pitfall | Symptom | Fix | Enforced by |
|---|---|---|---|---|
| 21 | `add_area` without `connect_adjacent` | solid, full-height wall between neighbouring areas | call `connect_adjacent(a, b)` for a two-sided portal | `DoomGen2/README.md`, snapshot |
| 22 | `add_boundary` doors on some Voronoi layouts | unreliable doors | build explicit door sectors (`DoomGen2/examples/coop_relay.py`) | example / snapshot |
| 23 | `-deathmatch` without deathmatch starts (editor number 11), or multi-player `-deathmatch` relying on raw deathmatch starts alone | missing starts crash on load (signal 11 / exit 139) with no console error; with starts present, ZDoom 2.8.1's `G_CheckSpot` rejects `type=0` deathmatch starts and ignores `+sv_spawnfarthest 1` at level load, picking starts with replacement so players telefrag on tic 0 even with `N+` starts | add at least one deathmatch start (11), **and** for telefrag-free multi-player spawns either (a) subclass `DoomPlayer` in `DECORATE` with `+NODAMAGE` + a state calling `A_ChangeFlag("NODAMAGE", FALSE)`, relocate players in `ENTER` to distinct `MapSpot`s via `SetActorPosition`, and clear `NODAMAGE` after settling (`DoomGen2/examples/simple_tag.py`), or (b) omit `-deathmatch` and use `Player 1..N` starts (`1..4`) | `validate_scenario.py` (static + load + contract), `capture_multiagent_scenario.py` (`overview.png` / `summary.json`) |
