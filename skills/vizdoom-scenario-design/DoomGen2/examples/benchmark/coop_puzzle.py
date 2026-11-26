import os
import random
from typing import Any, Dict

from doomgen.abstraction.layout import ConnectionType
from doomgen.batch.scenario import Scenario
from doomgen.builder import ProceduralMapBuilder
from doomgen.doom.things import ThingType
from doomgen.geometry.shapes import rect
from doomgen.logic.acs_builder import ACSBuilder, ScriptType


class CoopPuzzleScenario(Scenario):
    DOOR_DEPTH = 16

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "seed": 42,
            "num_zones": 10,
            "zone_length": 384,
            "lane_height": 240,
            "lane_gap": 0,
            "plate_size": 72,
            "plates_per_side": 2,
            "and_frequency_pairs": 3,
            "floor_texture": "FLOOR4_8",
            "ceiling_texture": "CEIL3_5",
            "wall_texture": "STONE2",
            "door_texture": "BIGDOOR2",
            "window_texture": "MIDGRATE",
            "plate_texture": "CEIL5_1",
            "highlight_texture_a": "AQF044",
            "highlight_texture_b": "AQF038",
            "light_level": 224,
        }

    def validate_config(self) -> None:
        num_zones = int(self.config["num_zones"])
        if num_zones < 2:
            raise ValueError("num_zones must be >= 2")
        if num_zones > 50:
            raise ValueError("num_zones must be <= 50 to match the ACS telemetry contract")
        if num_zones % 2 != 0:
            raise ValueError("num_zones must be even because coop_puzzle alternates room and hallway zones")
        if int(self.config["plates_per_side"]) < 1:
            raise ValueError("plates_per_side must be >= 1")
        if int(self.config["and_frequency_pairs"]) < 1:
            raise ValueError("and_frequency_pairs must be >= 1")
        if int(self.config["lane_gap"]) < 0:
            raise ValueError("lane_gap must be >= 0")
        plate_size = self._plate_size()
        if plate_size < 16:
            raise ValueError("plate_size must be >= 16")
        if plate_size >= int(self.config["lane_height"]):
            raise ValueError("plate_size must be smaller than lane_height")
        if plate_size >= int(self.config["zone_length"]):
            raise ValueError("plate_size must be smaller than zone_length")

    def _choose_seed(self) -> int:
        seed = self.config["seed"]
        return int(seed) if seed is not None else random.randint(0, 999999)

    def _zone_name(self, zone_idx: int, lane: str) -> str:
        return f"Zone_{zone_idx}_{lane}"

    def _pair_idx(self, zone_idx: int) -> int:
        return (zone_idx + 1) // 2

    def _zone_tag(self, zone_idx: int) -> int:
        return 200 + zone_idx

    def _plate_size(self) -> int:
        return int(self.config.get("plate_size", self.config.get("plate_width", 72)))

    def _type1_plate_tag(self, zone_idx: int, lane: str, offset: int) -> int:
        base = zone_idx * 1000 + (1 if lane == "A" else 101)
        return base + offset

    def _sync_plate_tag(self, zone_idx: int) -> int:
        return 100 + zone_idx

    def _lane_door_tag(self, pair_idx: int, lane: str) -> int:
        return pair_idx if lane == "A" else pair_idx + 50

    def _zone_center_x(self, zone_idx: int, zone_length: int) -> int:
        return (zone_idx - 1) * zone_length

    def _lane_center_y(self, lane: str) -> int:
        lane_height = int(self.config["lane_height"])
        lane_gap = int(self.config["lane_gap"])
        center = (lane_height + lane_gap) // 2
        return center if lane == "A" else -center

    def _plate_center_y(self, lane: str) -> int:
        lane_height = int(self.config["lane_height"])
        plate_size = self._plate_size()
        offset = max(0, (lane_height - plate_size) // 4)
        lane_center = self._lane_center_y(lane)
        return lane_center + offset if lane == "A" else lane_center - offset

    def _is_sync_zone(self, zone_idx: int) -> bool:
        num_zones = int(self.config["num_zones"])
        freq = int(self.config["and_frequency_pairs"])
        if zone_idx >= num_zones or zone_idx % 2 == 0:
            return False
        pair_idx = self._pair_idx(zone_idx)
        return pair_idx > freq and (pair_idx - (freq + 1)) % freq == 0

    def _make_builder(self, seed: int) -> ProceduralMapBuilder:
        zone_length = int(self.config["zone_length"])
        lane_height = int(self.config["lane_height"])
        lane_gap = int(self.config["lane_gap"])
        num_zones = int(self.config["num_zones"])
        pad = 640
        max_half_h = (lane_height + lane_gap + lane_height) // 2
        start_x = -(zone_length // 2)
        end_x = ((num_zones - 1) * zone_length) + (zone_length // 2)
        bounds = (start_x - pad, -(max_half_h + pad), end_x + pad, max_half_h + pad)
        return ProceduralMapBuilder(bounds=bounds, seed=seed)

    def _make_sync_zone_function(self, num_zones: int) -> str:
        lines = ["function int IsSyncZone(int zone_idx) {"]
        first = True
        for zone_idx in range(1, num_zones + 1):
            if not self._is_sync_zone(zone_idx):
                continue
            prefix = "    if" if first else "    else if"
            lines.append(f"{prefix} (zone_idx == {zone_idx}) return 1;")
            first = False
        lines.append("    return 0;")
        lines.append("}")
        return "\n".join(lines)

    def _make_acs(self, num_zones: int) -> ACSBuilder:
        acs = ACSBuilder()
        acs.add_include("zcommon.acs")
        acs.add_global_var("zone", 1, "int")
        acs.add_global_var("plate_state", 2, "int")
        acs.add_global_var("room_type", 3, "int")
        acs.add_global_var("zone_p1", 4, "int")
        acs.add_global_var("zone_p2", 5, "int")
        acs.add_global_var("zone_limit", 6, "int")
        acs.add_global_code(
            f"""
int active_plate[{num_zones + 1}];
int plate_in_b_zone[{num_zones + 1}];
int sync_timer[{num_zones + 1}];

function int cnt_sectors(int min_tag, int max_tag)
{{
    int count = 0;
    int t;
    for (t = min_tag; t <= max_tag; t++)
    {{
        if (GetSectorLightLevel(t) != -1) count++;
    }}
    return count;
}}

function int get_sector_tag(int min_tag, int max_tag, int n)
{{
    int count = 0;
    int t;
    for (t = min_tag; t <= max_tag; t++)
    {{
        if (GetSectorLightLevel(t) != -1)
        {{
            count++;
            if (count == n) return t;
        }}
    }}
    return -1;
}}

{self._make_sync_zone_function(num_zones)}
""".strip()
        )
        acs.add_script(
            ScriptType.OPEN,
            f"""
int i;
int pos;
int iidx;
int cnt_a;
int cnt_b;
int chosen;
int rand;
int next_zone;

for (i = 1; i <= {num_zones}; i += 2)
{{
    active_plate[i] = 0;
    plate_in_b_zone[i] = 0;
    next_zone = i + 1;
    if (next_zone <= {num_zones})
    {{
        active_plate[next_zone] = 0;
        plate_in_b_zone[next_zone] = 0;
    }}

    if (IsSyncZone(i)) continue;

    rand = Random(0, 1);

    iidx = i * 1000;
    cnt_a = cnt_sectors(iidx + 1, iidx + 99);
    cnt_b = cnt_sectors(iidx + 101, iidx + 199);

    if (cnt_a + cnt_b == 0) continue;

    if (rand == 0 && cnt_a > 0)
    {{
        pos = Random(1, cnt_a);
        chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
        active_plate[i] = chosen;
        plate_in_b_zone[i] = 0;
        ChangeFloor(chosen, "{self.config["highlight_texture_a"]}");
    }}
    else if (rand == 1 && cnt_b > 0)
    {{
        pos = Random(1, cnt_b);
        chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
        active_plate[i] = chosen;
        plate_in_b_zone[i] = 1;
        ChangeFloor(chosen, "{self.config["highlight_texture_b"]}");
    }}
    else if (cnt_a > 0)
    {{
        pos = Random(1, cnt_a);
        chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
        active_plate[i] = chosen;
        plate_in_b_zone[i] = 0;
        ChangeFloor(chosen, "{self.config["highlight_texture_a"]}");
    }}
    else
    {{
        pos = Random(1, cnt_b);
        chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
        active_plate[i] = chosen;
        plate_in_b_zone[i] = 1;
        ChangeFloor(chosen, "{self.config["highlight_texture_b"]}");
    }}

    if (next_zone > {num_zones}) continue;

    iidx = next_zone * 1000;
    cnt_a = cnt_sectors(iidx + 1, iidx + 99);
    cnt_b = cnt_sectors(iidx + 101, iidx + 199);

    if (cnt_a + cnt_b == 0) continue;

    if (rand == 0 && cnt_b > 0)
    {{
        pos = Random(1, cnt_b);
        chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
        active_plate[next_zone] = chosen;
        plate_in_b_zone[next_zone] = 1;
        ChangeFloor(chosen, "{self.config["highlight_texture_b"]}");
    }}
    else if (rand == 1 && cnt_a > 0)
    {{
        pos = Random(1, cnt_a);
        chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
        active_plate[next_zone] = chosen;
        plate_in_b_zone[next_zone] = 0;
        ChangeFloor(chosen, "{self.config["highlight_texture_a"]}");
    }}
    else if (cnt_b > 0)
    {{
        pos = Random(1, cnt_b);
        chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
        active_plate[next_zone] = chosen;
        plate_in_b_zone[next_zone] = 1;
        ChangeFloor(chosen, "{self.config["highlight_texture_b"]}");
    }}
    else
    {{
        pos = Random(1, cnt_a);
        chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
        active_plate[next_zone] = chosen;
        plate_in_b_zone[next_zone] = 0;
        ChangeFloor(chosen, "{self.config["highlight_texture_a"]}");
    }}
}}
""".strip(),
            number=2,
        )
        acs.add_script(
            ScriptType.OPEN,
            f"""
int new_zone;
int plate;
int z;
int i;
int p1_on;
int p2_on;
int door_pair;
int plate_tag;
int door_tag;
int sync_active;
int p1_zone;
int p2_zone;

Delay(5);

SetActivator(0, AAPTR_PLAYER1);
Thing_ChangeTID(0, 1);
SetActivator(0, AAPTR_PLAYER2);
Thing_ChangeTID(0, 2);
SetActivator(0, AAPTR_NULL);

zone = 0;
plate_state = 0;
room_type = 0;
zone_p1 = 0;
zone_p2 = 0;
zone_limit = {num_zones};

while (true)
{{
    new_zone = 0;
    plate = 0;
    p1_zone = 0;
    p2_zone = 0;

    for (z = 201; z <= {200 + num_zones}; z++)
    {{
        if (ThingCountSector(T_NONE, 1, z) > 0) p1_zone = z - 200;
        if (ThingCountSector(T_NONE, 2, z) > 0) p2_zone = z - 200;
    }}

    if (p1_zone > new_zone) new_zone = p1_zone;
    if (p2_zone > new_zone) new_zone = p2_zone;

    for (i = 1; i <= {num_zones}; i++)
    {{
        door_pair = (i + 1) / 2;

        if (IsSyncZone(i))
        {{
            plate_tag = 100 + i;
            p1_on = ThingCountSector(T_NONE, 1, plate_tag) > 0;
            p2_on = ThingCountSector(T_NONE, 2, plate_tag) > 0;
            if (p1_on && i > p1_zone) p1_zone = i;
            if (p2_on && i > p2_zone) p2_zone = i;
            sync_active = (p1_on && p2_on);

            if (sync_active)
            {{
                plate = 1;
                sync_timer[i] = 105;
            }}

            if (sync_timer[i] > 0)
            {{
                Door_Open(door_pair, 64, 0);
                Door_Open(door_pair + 50, 64, 0);
            }}
            else
            {{
                Door_Close(door_pair, 64, 0);
                Door_Close(door_pair + 50, 64, 0);
            }}

            if (sync_timer[i] > 0) sync_timer[i]--;
            continue;
        }}

        plate_tag = active_plate[i];
        if (plate_tag == 0) continue;

        p1_on = ThingCountSector(T_NONE, 1, plate_tag) > 0;
        p2_on = ThingCountSector(T_NONE, 2, plate_tag) > 0;
        if (p1_on && i > p1_zone) p1_zone = i;
        if (p2_on && i > p2_zone) p2_zone = i;

        if (p1_on || p2_on)
        {{
            plate = 1;
        }}

        if (plate_in_b_zone[i] == 1)
        {{
            door_tag = door_pair;
        }}
        else
        {{
            door_tag = door_pair + 50;
        }}

        if (p1_on || p2_on)
        {{
            Door_Open(door_tag, 64, 0);
        }}
        else
        {{
            Door_Close(door_tag, 64, 0);
        }}
    }}

    if (p1_zone > zone_p1) zone_p1 = p1_zone;
    if (p2_zone > zone_p2) zone_p2 = p2_zone;
    if (zone_p1 > new_zone) new_zone = zone_p1;
    if (zone_p2 > new_zone) new_zone = zone_p2;

    if (new_zone > 0)
    {{
        zone = new_zone;
        room_type = IsSyncZone(new_zone);
    }}
    plate_state = plate;

    if (
        ThingCountSector(T_NONE, 1, {200 + num_zones}) > 0
        && ThingCountSector(T_NONE, 2, {200 + num_zones}) > 0
    )
    {{
        Exit_Normal(0);
    }}

    Delay(1);
}}
""".strip(),
            number=1,
        )
        return acs

    def _build_components(self, seed: int):
        cfg = self.config
        num_zones = int(cfg["num_zones"])
        zone_length = int(cfg["zone_length"])
        lane_height = int(cfg["lane_height"])
        plate_size = self._plate_size()
        plates_per_side = int(cfg["plates_per_side"])
        lane_gap = int(cfg["lane_gap"])

        floor_texture = cfg["floor_texture"]
        ceiling_texture = cfg["ceiling_texture"]
        wall_texture = cfg["wall_texture"]
        door_texture = cfg["door_texture"]
        window_texture = cfg["window_texture"]
        plate_texture = cfg["plate_texture"]
        light_level = int(cfg["light_level"])

        builder = self._make_builder(seed)
        acs = self._make_acs(num_zones)
        metadata: Dict[str, Any] = {"zones": {}, "relay_doors": {}, "sync_zones": {}}

        for zone_idx in range(1, num_zones + 1):
            x_pos = self._zone_center_x(zone_idx, zone_length)
            zone_tag = self._zone_tag(zone_idx)
            metadata["zones"][zone_idx] = {"tag": zone_tag, "type1": {"A": [], "B": []}}

            for lane in ("A", "B"):
                zone_name = self._zone_name(zone_idx, lane)
                builder.add_area(
                    zone_name,
                    shape=rect(x_pos, self._lane_center_y(lane), zone_length, lane_height),
                    floor_height=0,
                    ceiling_height=128,
                    floor_texture=floor_texture,
                    ceiling_texture=ceiling_texture,
                    wall_texture=wall_texture,
                    light_level=light_level,
                    tag=zone_tag,
                )
                metadata["zones"][zone_idx][lane] = zone_name

            window_a = self._zone_name(zone_idx, "A")
            if lane_gap > 0:
                window_a = f"LaneDivider_{zone_idx}"
                builder.add_area(
                    window_a,
                    shape=rect(x_pos, 0, zone_length, lane_gap),
                    floor_height=0,
                    ceiling_height=128,
                    floor_texture=floor_texture,
                    ceiling_texture=ceiling_texture,
                    wall_texture=wall_texture,
                    light_level=light_level,
                )
                builder.connect_adjacent(
                    self._zone_name(zone_idx, "A"),
                    window_a,
                    ConnectionType.WINDOW,
                    middle_texture=window_texture,
                    blocking=True,
                )
            builder.connect_adjacent(
                window_a,
                self._zone_name(zone_idx, "B"),
                ConnectionType.WINDOW,
                middle_texture=window_texture,
                blocking=True,
            )

            if zone_idx == 1:
                start_x = x_pos - (zone_length // 2) + 80
                builder.add_thing(ThingType.PLAYER1_START, start_x, self._lane_center_y("A"), angle=0)
                builder.add_thing(ThingType.PLAYER2_START, start_x, self._lane_center_y("B"), angle=0)

            plate_spacing = zone_length // (plates_per_side + 1)
            if self._is_sync_zone(zone_idx):
                sync_tag = self._sync_plate_tag(zone_idx)
                metadata["sync_zones"][zone_idx] = {"tag": sync_tag, "plates": {}}
                sync_x = x_pos - (zone_length // 2) + plate_spacing
                for lane in ("A", "B"):
                    plate_name = f"SyncPlate_{zone_idx}_{lane}"
                    plate_y = self._plate_center_y(lane)
                    sync_floor_texture = self.config["highlight_texture_a"] if lane == "A" else self.config["highlight_texture_b"]
                    builder.add_area(
                        plate_name,
                        shape=rect(sync_x, plate_y, plate_size, plate_size),
                        mode="overwrite",
                        floor_height=0,
                        ceiling_height=128,
                        floor_texture=sync_floor_texture,
                        ceiling_texture=ceiling_texture,
                        wall_texture=wall_texture,
                        light_level=light_level,
                        tag=sync_tag,
                    )
                    builder.connect_adjacent(self._zone_name(zone_idx, lane), plate_name, ConnectionType.OPEN, silent=False)
                    metadata["sync_zones"][zone_idx]["plates"][lane] = {
                        "name": plate_name,
                        "tag": sync_tag,
                        "x": sync_x,
                        "y": plate_y,
                        "size": plate_size,
                    }
            else:
                for lane in ("A", "B"):
                    for offset in range(plates_per_side):
                        plate_x = x_pos - (zone_length // 2) + plate_spacing * (offset + 1)
                        plate_y = self._plate_center_y(lane)
                        plate_tag = self._type1_plate_tag(zone_idx, lane, offset)
                        plate_name = f"Plate_{zone_idx}_{lane}_{offset + 1}"
                        builder.add_area(
                            plate_name,
                            shape=rect(plate_x, plate_y, plate_size, plate_size),
                            mode="overwrite",
                            floor_height=0,
                            ceiling_height=128,
                            floor_texture=plate_texture,
                            ceiling_texture=ceiling_texture,
                            wall_texture=wall_texture,
                            light_level=light_level,
                            tag=plate_tag,
                        )
                        builder.connect_adjacent(self._zone_name(zone_idx, lane), plate_name, ConnectionType.OPEN, silent=False)
                        metadata["zones"][zone_idx]["type1"][lane].append(
                            {
                                "name": plate_name,
                                "tag": plate_tag,
                                "x": plate_x,
                                "y": plate_y,
                                "size": plate_size,
                            }
                        )

        for zone_idx in range(1, num_zones, 2):
            pair_idx = self._pair_idx(zone_idx)
            metadata["relay_doors"][pair_idx] = {}
            door_x = self._zone_center_x(zone_idx, zone_length) + zone_length // 2
            for lane in ("A", "B"):
                door_name = f"RelayDoor_{pair_idx}_{lane}"
                door_tag = self._lane_door_tag(pair_idx, lane)
                builder.add_area(
                    door_name,
                    shape=rect(door_x, self._lane_center_y(lane), self.DOOR_DEPTH, lane_height),
                    mode="overwrite",
                    floor_height=0,
                    ceiling_height=0,
                    wall_texture=door_texture,
                    floor_texture=door_texture,
                    ceiling_texture=door_texture,
                    light_level=light_level,
                    tag=door_tag,
                )
                builder.connect_adjacent(self._zone_name(zone_idx, lane), door_name, ConnectionType.DOOR)
                builder.connect_adjacent(door_name, self._zone_name(zone_idx + 1, lane), ConnectionType.DOOR)
                metadata["relay_doors"][pair_idx][lane] = {"name": f"RelayDoor_{pair_idx}_{lane}", "tag": door_tag}

        for zone_idx in range(2, num_zones):
            if zone_idx % 2 != 0:
                continue
            builder.connect_adjacent(self._zone_name(zone_idx, "A"), self._zone_name(zone_idx + 1, "A"), ConnectionType.OPEN)
            builder.connect_adjacent(self._zone_name(zone_idx, "B"), self._zone_name(zone_idx + 1, "B"), ConnectionType.OPEN)

        return builder, acs, metadata

    def generate(self, output_path: str) -> None:
        seed = self._choose_seed()
        random.seed(seed)
        print(f"Generating Co-op Puzzle Benchmark with {self.config['num_zones']} zones (Seed: {seed})...")

        builder, acs, _ = self._build_components(seed)

        try:
            print("Compiling ACS...")
            bytecode = acs.compile()
            if not bytecode:
                raise RuntimeError("No bytecode generated")
            builder.map_data.behavior = bytecode
            print("ACS compiled successfully.")
        except Exception as exc:
            print(f"ACS compilation failed: {exc}")
            print("Using dummy BEHAVIOR lump. GZDoom will compile SCRIPTS on load.")
            builder.map_data.behavior = b"ACSE\x08\x00\x00\x00\x00\x00\x00\x00"

        builder.map_data.scripts = acs.to_code()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        builder.build(output_path)
        print(f"WAD written to {output_path}")


def create_coop_puzzle() -> None:
    scenario = CoopPuzzleScenario(name="coop_puzzle")
    scenario.generate("examples/benchmark/output/coop_puzzle.wad")


if __name__ == "__main__":
    create_coop_puzzle()
