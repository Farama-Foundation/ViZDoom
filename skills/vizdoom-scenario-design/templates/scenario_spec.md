# Scenario spec: `<scenario_name>`

Fill this in before building geometry or writing ACS. The `USER` variables and
the termination rules are what the validation script and the Python wrapper test against.
Replace every `<...>`; delete rows that do not apply. This template is optional.

## 1. Purpose

- Capability under test: `<e.g. localising a target by sound>`
- Control / variant that isolates it: `<e.g. same map with a visible target>`
- Chance or trivial baseline: `<e.g. 50 % for a two-way choice>`

## 2. Players and roles

| Role | Count | How the role is bound | Abilities / restrictions | Appearance |
|---|---|---|---|---|
| `<hunter>` | `<3>` | `<player name Player0-2 (not network slot)>` | `<speed 0.75, no weapons>` | `<red translation>` |
| `<prey>` | `<1>` | `<player name Player3>` | `<speed 1.0, invisible, chirps every 10 tics>` | `<STYLE_None, alpha 0>` |

- Multiplayer mode: `<single player | cooperative netgame | -deathmatch>`
- Respawn: `<none: +sv_norespawn 1 | after N tics>`

## 3. Map

- Route: `<DoomGen generator | patch of existing WAD | hand-made in SLADE/UDB>`
- Layout: `<size in map units, rooms, obstacles, hazards>`
- Spawns: `<player/deathmatch starts, MapSpot tids, per-episode rotation>`
- Things: `<class names from references/actor_classes.md>`
- Custom lumps: `<DECORATE actors (parents checked), SNDINFO + sounds>`

## 4. `USER` variables (ACS globals exported to Python)

All exported globals are reset in the `OPEN` script, so the variables holds however the map is
restarted (single-player new game or netgame map change).

| Variable | ACS global | Type | Value before / after the event | Reset |
|---|---|---|---|---|
| `USER<50>` | `global int <50>:<caught>` | `<int flag>` | `<0 until the first catch, then latched 1>` | `OPEN` |
| `USER<51>` | `global int <51>:<catcher>` | `<int>` | `<0, then catcher index + 1>` | `OPEN` |
| `USER<52>` | `global int <52>:<elapsed>` | `<int tics>` | `<active tics, frozen on catch>` | `OPEN` |

## 5. Reward and termination

- ACS reward (`global int 0`, fixed point): `<none | +1.0 per ...>`
- Engine reward terms in the cfg: `<living_reward, death_penalty, ...>`
- Wrapper reward: `<+10 per hunter / -10 prey on USER50 = 1>`
- Terminates when: `<USER50 becomes 1>` (never on `HEALTH`/`DEAD`/frags unless that *is* the rule)
- Truncates when: `<episode_timeout = 4235 tics (35 settle + 4200 active)>`
- Settling period at map start: `<35 tics, players protected with NODAMAGE>`

## 6. Randomisation (per episode, in `OPEN`)

- `<spawn rotation by Random(0, 3) quarter turns>`
- `<target colour / position shuffle>`
- Note: netgame episodes do not re-seed the RNG; episodes differ but cannot be replayed by seed.

## 7. Observations and actions (`.cfg`)

- Buttons: `<MOVE_LEFT MOVE_RIGHT MOVE_FORWARD MOVE_BACKWARD TURN_LEFT TURN_RIGHT>`
- Game variables: `<HEALTH DEAD PLAYER_NUMBER USER50 USER51 USER52>`
- Buffers: `<screen 320x240; audio_buffer_enabled, audio_buffer_size = 4; notifications>`
- `game_args`: `<-deathmatch +sv_norespawn 1 +sv_noautoaim 1 ...>`

## 8. Acceptance tests

- [ ] `acc` compiles; every ACS call is in `references/acs_functions.md` / `line_specials.md`
- [ ] `validate_scenario.py --cfg <cfg> --players <N> --expect USER<50>=0 ...` passes
      (loads, variables holds after every reset, episode ends within the timeout)
- [ ] scenario-specific probes: `<e.g. one prey death after the latch; roles follow names under shuffled joins>`
- [ ] capture snapshot inspected (frames, top-down map, audio, notifications)
- [ ] human playtest with `play_netgame.sh` (`<what the tester should look for>`)
- [ ] `make -j` run, and the package copy is what training loads
