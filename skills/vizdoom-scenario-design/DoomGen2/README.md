# DoomGen2

A Python library for procedurally generating Doom WAD files for single- and multi-agent
reinforcement learning.

Walls keep their **real shapes**: rectangles, circles, rings and arbitrary polygons become
straight (or segmented round) linedefs following the exact outline. **Voronoi diagrams are
used for placement** — Lloyd-relaxed spots, clipped Voronoi cells, a Delaunay graph and its
minimum spanning tree decide where spawns, pickups and obstacles go while keeping the map
navigable.

## Installation

Vendored inside the `vizdoom-scenario-design` skill (upstream:
<https://github.com/TTomilin/COMRAD/tree/main/DoomGen2>). Use it in place:

```bash
pip install -e ".[dev]"      # from this folder, or: export PYTHONPATH=<this folder>/src
pytest                       # runs tests/ with src/ and examples/ on the path
```

Example scripts write to `examples/output/` next to themselves (git-ignored) unless
`--output` is passed. `acc` (`acc/`) and `zdbsp` (`zdbsp/`) ship as macOS/Windows binaries;
on Linux build `acc` from <https://github.com/rheit/acc> and pass `--acc` (ZDBSP is skipped
automatically if unavailable and ViZDoom builds nodes at load time).

## Quick Start

```python
from doomgen.abstraction.layout import ConnectionType
from doomgen.builder import ProceduralMapBuilder
from doomgen.geometry.shapes import circle

builder = ProceduralMapBuilder(seed=42)
builder.add_area("Main Hall", shape=(0, 0, 512, 512), floor_texture="FLOOR4_8")
builder.add_area("Armory", shape=(384, 0, 256, 256), floor_texture="CEIL5_1")
builder.add_area("Cave", shape=circle(-900, 400, 256), floor_texture="NUKAGE1")

builder.add_boundary(
    "Main Hall", "Armory", name="Door1", connection_type=ConnectionType.DOOR,
    ceiling_height=0, door_width=128,
)
corridor = builder.add_corridor("Main Hall", "Cave", width=128)

crates = builder.scatter_obstacles(
    "Main Hall", 4, footprint=(64, 64), spacing=64,
    create_areas=True, floor_height=48, wall_texture="CRATE1",
)
(start, goal, _dist), = builder.pick_spawn_pairs(
    ["Main Hall", corridor, "Cave"], 1, min_path_distance=800, obstacles=crates
)
builder.add_thing("PLAYER1_START", *start, angle=0)
builder.add_thing("SHOTGUN", *builder.get_random_point_in_area("Armory"))
builder.build("examples/output/my_map.wad")
```

## Architecture

- **`ProceduralMapBuilder`** (`doomgen/builder.py`): High-level facade for map generation and placement.
- **`PolygonLayout`** (`doomgen/geometry/arrangement.py`): Resolves layered area polygons into regions and classified wall edges (`void_wall` / `solid` / `portal`).
- **Shapes and routing** (`doomgen/geometry/shapes.py`, `routing.py`): `rect`, `circle`, `ring`, `regular_polygon`; straight / L / grid corridor routes and stairs.
- **`doomgen.placement`**: Lloyd relaxation, clipped Voronoi cells, obstacle scattering, Delaunay + MST navigation graphs, max-clearance spots.
- **`MapTranslator`** (`doomgen/doom/translator.py`): Converts edges and the area graph into Doom sectors, sidedefs and linedefs.
- **`ACSBuilder`** (`doomgen/logic/acs_builder.py`): Generates and compiles ACS bytecode via `acc`.
- **`Scenario` / `BatchGenerator`** (`doomgen/batch/`): Parameterised scenario generation and grid sweeps.

Touching areas are separated by solid walls until connected with `connect_adjacent` (or
`auto_connect_touching`); connected edges become two-sided portals with lower/upper
textures so flush sectors can be raised later by ACS.

## Requirements

- Python 3.10+
- `omgifol`, `shapely`, `numpy`, `scipy`, `networkx`
- Optional: `matplotlib` (for `visualize_layout`)
- `acc` (ACS Compiler) — bundled in `acc/` for macOS/Windows; see Installation for Linux.

## Examples

Five self-contained examples live in `examples/` (full index in `examples/README.md`):

```bash
PYTHONPATH=src python examples/geometry_and_placement.py
PYTHONPATH=src python examples/key_maze.py
PYTHONPATH=src python examples/coop_relay.py
PYTHONPATH=src python examples/simple_tag.py --audio --output /tmp/simple_tag_audio.wad
PYTHONPATH=src python examples/deathmatch_arena.py --batch-dir /tmp/arena_batch
```
