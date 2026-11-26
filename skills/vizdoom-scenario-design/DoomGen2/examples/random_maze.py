import os
import random
from doomgen.builder import ProceduralMapBuilder
from doomgen.abstraction.layout import ConnectionType
from doomgen.doom.things import ThingType

def create_maze_level():
    print("Generating Random Maze Level...")

    # 1. Initialize Builder
    seed = random.randint(0, 999999)
    print(f"Seed: {seed}")
    # Tighter bounds for higher resolution with smaller rooms
    builder = ProceduralMapBuilder(bounds=(-1500, -1500, 1500, 1500), seed=seed)

    # 2. Grid Parameters
    GRID_W = 8
    GRID_H = 8
    CELL_SIZE = 256
    SPACING = 256 # Rooms touch exactly
    OFFSET_X = - (GRID_W * SPACING) / 2
    OFFSET_Y = - (GRID_H * SPACING) / 2

    # 3. Generate Rooms
    rooms = {} # (x,y) -> area_id

    for x in range(GRID_W):
        for y in range(GRID_H):
            pos_x = OFFSET_X + x * SPACING
            pos_y = OFFSET_Y + y * SPACING

            name = f"Room_{x}_{y}"

            # Determine Room Type
            floor_tex = "FLOOR4_8" # Default Grey
            ceil_tex = "CEIL3_5"

            if x == 0 and y == 0:
                floor_tex = "FLAT14" # Start (Greenish)
            elif x == GRID_W - 1 and y == GRID_H - 1:
                floor_tex = "CEIL5_1" # End (Reddish)

            area_id = builder.add_area(
                name=name,
                shape=(pos_x, pos_y, CELL_SIZE, CELL_SIZE),
                floor_height=0,
                ceiling_height=128,
                floor_texture=floor_tex,
                ceiling_texture=ceil_tex,
                wall_texture="STARTAN3"
            )
            rooms[(x,y)] = area_id

            # Add Player Start at (0,0)
            if x == 0 and y == 0:
                builder.add_thing("PLAYER1_START", pos_x, pos_y, angle=90)

    # 4. Generate Maze (Recursive Backtracker)
    visited = set()
    stack = [(0,0)]
    visited.add((0,0))

    # Directions: N, E, S, W
    directions = [(0,1), (1,0), (0,-1), (-1,0)]

    # Store connections to make: ((x1,y1), (x2,y2))
    connections = []

    while stack:
        cx, cy = stack[-1]

        # Find unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            connections.append(((cx, cy), (nx, ny)))
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    # 5. Apply Connections & Locks
    # We want to lock the entrance to the Exit Room (GRID_W-1, GRID_H-1)
    exit_pos = (GRID_W-1, GRID_H-1)
    start_pos = (0,0)

    # Build Adjacency Map for Pathfinding
    adj = {}
    for p1, p2 in connections:
        if p1 not in adj: adj[p1] = []
        if p2 not in adj: adj[p2] = []
        adj[p1].append(p2)
        adj[p2].append(p1)

    # Helper to find path
    def get_path(start, end):
        q = [(start, [start])]
        visited_bfs = {start}
        while q:
            curr, path = q.pop(0)
            if curr == end:
                return path
            for n in adj.get(curr, []):
                if n not in visited_bfs:
                    visited_bfs.add(n)
                    q.append((n, path + [n]))
        return None

    # Find valid Key Location
    # Key must be reachable from Start WITHOUT passing through Exit
    possible_key_locs = [k for k in rooms.keys() if k != start_pos and k != exit_pos]
    random.shuffle(possible_key_locs)

    key_pos = None
    for k in possible_key_locs:
        path = get_path(start_pos, k)
        if path and exit_pos not in path:
            key_pos = k
            break

    if not key_pos:
        print("Warning: Could not find valid key location! Defaulting to random (might be broken).")
        key_pos = possible_key_locs[0]

    # Place Red Key
    kx, ky = key_pos
    k_px = OFFSET_X + kx * SPACING
    k_py = OFFSET_Y + ky * SPACING
    builder.add_thing("RED_KEYCARD", k_px, k_py)
    print(f"Red Key placed at Room_{kx}_{ky}")

    # Process Connections
    for (p1, p2) in connections:
        room_a = f"Room_{p1[0]}_{p1[1]}"
        room_b = f"Room_{p2[0]}_{p2[1]}"

        # Check if this is the connection to the Exit
        is_exit_conn = (p1 == exit_pos or p2 == exit_pos)

        if is_exit_conn:
            # Create Red Door
            # Hexen Special 13: Door_LockedRaise(tag, speed, delay, lockid)
            # LockID 1 = Red Card
            print(f"Creating Red Door between {room_a} and {room_b}")
            builder.add_boundary(
                room_a, room_b,
                name=f"Door_{room_a}_{room_b}",
                connection_type=ConnectionType.DOOR,
                floor_height=0, ceiling_height=0, # Closed
                wall_texture="DOORRED", # Visual cue
                linedef_special=13, # Door_LockedRaise
                linedef_args=[0, 16, 150, 1, 0], # tag=0 (self), speed=16, delay=150, lock=1 (Red Card)
                linedef_flags=0x600, # 0x200 (Repeatable) | 0x400 (Player Use)
                door_width=128, # Rest of the shared edge stays solid wall
                door_depth=16
            )
        else:
            # Standard Connection
            # Since rooms touch, we can use connect_adjacent or add_boundary(OPEN)
            # Let's use add_boundary(OPEN) to create a "doorway" look (lintel)
            # Or just connect_adjacent for open space.
            # Let's use connect_adjacent for open flow.
            builder.connect_adjacent(room_a, room_b, ConnectionType.OPEN)

    # 6. Add Enemies and Decor
    for x in range(GRID_W):
        for y in range(GRID_H):
            if (x,y) == (0,0): continue # Skip start

            pos_x = OFFSET_X + x * SPACING
            pos_y = OFFSET_Y + y * SPACING

            # Random chance for enemies
            if random.random() < 0.4:
                enemy = random.choice(["IMP", "SHOTGUN_GUY", "ZOMBIEMAN"])
                builder.add_thing(enemy, pos_x, pos_y)

            # Random chance for health/ammo
            if random.random() < 0.3:
                item = random.choice(["MEDIKIT", "SHELLS", "CLIP"])
                # Offset slightly
                builder.add_thing(item, pos_x + 32, pos_y + 32)

    # 7. Build
    # We need a dummy behavior for Hexen format to work (so we can use Special 13)
    builder.map_data.behavior = b'ACSE\x08\x00\x00\x00\x00\x00\x00\x00'

    output_path = "examples/output/random_maze.wad"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    builder.build(output_path)
    print(f"WAD written to {output_path}")

if __name__ == "__main__":
    create_maze_level()
