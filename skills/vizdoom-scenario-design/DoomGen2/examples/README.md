# DoomGen2 examples

Five self-contained generators covering the main geometry, placement and ViZDoom
scenario patterns. Every script accepts `--output <path.wad>` (defaulting to
`examples/output/<name>.wad`, git-ignored).

| File | Patterns demonstrated |
|---|---|
| `geometry_and_placement.py` | Exact shapes (`rect`, `circle`), damaging floor, `connect_adjacent`, `add_corridor`, `add_stairs`, `ConnectionType.DOOR`, Voronoi `scatter_obstacles`, `nav_graph`, `pick_spawn_pairs`, `get_spots`, `clearance_spots`, optional `--plot` |
| `key_maze.py` | Seedable grid maze (`random.Random`), reachability-checked key placement, locked door (`Door_LockedRaise`), compiled ACS with fixed-point `global int 0:reward` |
| `coop_relay.py` | 2-player coop lanes with grated window (`MIDGRATE`), explicit door sectors (`Door_Open`), `ClearInventory`, private HUD clue (`SetHudSize` + `HUDMSG_LOG`), `OPEN` randomisation, `USER40`-`USER44` contract |
| `simple_tag.py` | 4-player `-deathmatch` arena, `mode="overwrite"` obstacles, `DECORATE` + `MAPINFO`, name-bound roles, `CreateTranslation`, `CheckSight`, latched `USER50`-`USER52` contract; `--audio` adds invisible prey, `SNDINFO`, synthetic WAV lump and `PlaySound` |
| `deathmatch_arena.py` | `Scenario` + `BatchGenerator` (`--batch-dir`), per-episode raised cover (`Floor_MoveToValue`), custom `DECORATE` target with `ACS_ExecuteAlways` death hook, safe literal `SpawnSpot` + `ThingCount`, fixed-point reward |

Run any example from any working directory:

```sh
PYTHONPATH=skills/vizdoom-scenario-design/DoomGen2/src \
    python skills/vizdoom-scenario-design/DoomGen2/examples/simple_tag.py --audio --output /tmp/simple_tag_audio.wad
```
