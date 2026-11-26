# DoomGen

A Python library for procedurally generating Doom WAD files, specifically designed for creating benchmarks for **Multi-Agent Reinforcement Learning (MARL)**.

Walls keep their **real shapes**: rectangles, circles, rings and arbitrary polygons become straight (or finely segmented round) linedefs that follow the exact outline. **Voronoi diagrams are used only for placement** — Lloyd-relaxed spots, clipped Voronoi cells, a Delaunay graph and its minimum spanning tree decide where spawns, pickups and obstacles go, with clearance and spacing rules that keep the map navigable (following *Beyond Specialization: Robust RL Navigation via Procedural Map Generators*, arXiv 2605.02528).

## Features

- **Exact Polygon Geometry**: Areas are layered polygons (`claim_void` fills empty space, `overwrite` cuts inner sectors, doors and pillars); all outlines are noded into Doom sectors.
- **Logical Layouts**: Define areas (rooms) and connections (doors, open passages) abstractly.
- **Corridors and Stairs**: Straight corridors when rooms line up, otherwise a single L bend, with a grid detour around other areas as a fallback. Widths are in map units.
- **Voronoi Placement**: Evenly spread, wall-safe spawn spots; obstacles scattered one per Voronoi cell with a guaranteed player-sized gap; navigation graphs with path-distance spawn pairs; max-clearance (medial-axis) spots.
- **ACS Scripting**: Inject game logic (puzzles, rewards, state tracking) via a Pythonic ACS builder.
- **Deterministic**: Fully seedable for reproducible RL benchmarks.
- **Doom 2 Compatible**: Generates standard UDMF or Hexen format WADs (via `omgifol`).

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mitkozh/doomgen.git
cd doomgen
pip install -e ".[dev]"
```

## Quick Start

```python
from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType

from doomgen.geometry.shapes import circle

# 1. Initialize Builder with a seed (bounds are optional and only clip shapes)
builder = ProceduralMapBuilder(seed=42)

# 2. Define Areas with exact shapes: (cx, cy, w, h) tuples or shapely polygons
builder.add_area("Main Hall", shape=(0, 0, 512, 512), floor_texture="FLOOR4_8")
builder.add_area("Armory", shape=(384, 0, 256, 256), floor_texture="CEIL5_1")
builder.add_area("Cave", shape=circle(-900, 400, 256), floor_texture="NUKAGE1")

# 3. Connect Areas
# A 128-unit door rectangle straddling the shared edge
builder.add_boundary("Main Hall", "Armory", name="Door1", connection_type=ConnectionType.DOOR,
                     ceiling_height=0, door_width=128)
# A straight or L-shaped corridor, 128 map units wide
corridor = builder.add_corridor("Main Hall", "Cave", width=128)

# 4. Place things with Voronoi-based placement
crates = builder.scatter_obstacles("Main Hall", 4, footprint=(64, 64), spacing=64,
                                   create_areas=True, floor_height=48, wall_texture="CRATE1")
# Start/goal pair at least 800 units apart along the navigation graph
(start, goal, _dist), = builder.pick_spawn_pairs(["Main Hall", corridor, "Cave"], 1,
                                                 min_path_distance=800, obstacles=crates)
builder.add_thing("PLAYER1_START", *start, angle=0)
builder.add_thing("SHOTGUN", *builder.get_random_point_in_area("Armory"))

# 5. Build WAD
builder.build("output/my_map.wad")
```

## Architecture

- **`ProceduralMapBuilder`** (`doomgen/builder.py`): High-level facade for map generation and placement.
- **`PolygonLayout`** (`doomgen/geometry/arrangement.py`): Resolves layered area polygons into regions and classified, oriented wall edges (`void_wall` / `solid` / `portal`).
- **Shapes and routing** (`doomgen/geometry/shapes.py`, `routing.py`): `rect`, `circle`, `ring`, `regular_polygon`; straight / L / grid corridor routes.
- **`doomgen.placement`**: Lloyd relaxation, clipped Voronoi cells, obstacle scattering, Delaunay + MST navigation graphs, max-clearance spots.
- **`MapTranslator`**: Converts the edge list plus the area graph into Doom sectors, sidedefs and linedefs.
- **`ACSBuilder`**: Generates Action Code Script (ACS) bytecode for map logic.

Touching areas are separated by solid walls until they are connected with `connect_adjacent` (or `auto_connect_touching`); connected edges become two-sided portals with lower/upper textures, so flush sectors can be raised later by ACS.

Run `python examples/test_placement.py` to generate a demo map plus a PNG of the layout, spots, obstacles and navigation graph.

## Requirements

- Python 3.10+
- `shapely` (geometry), `scipy`, `numpy`, `networkx` (placement)
- `omgifol` (for WAD I/O)
- `acc` (ACS Compiler) - usually included with GZDoom or available separately.

## Benchmark Scenarios

Run the new chained-platform benchmark:

```bash
python examples/benchmark/platform_chain.py
```

This generates:

- `examples/benchmark/output/platform_chain.wad`

Batch-generate varied platform-chain configurations:

```bash
python examples/benchmark/batches/generate_platform_chain_batch.py
```

Batch output directory:

- `examples/benchmark/output/batch_platform_chain/`
