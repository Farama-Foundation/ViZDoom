"""WAD file creation and manipulation via omgifol."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Any
from pathlib import Path

try:
    from omg import WAD, MapEditor as OMGMapEditor
    from omg.mapedit import Vertex, Linedef, Sidedef, Sector, Thing
    from omg.mapedit import ZLinedef, ZThing  # Hexen format for ACS support
    OMGIFOL_AVAILABLE = True
except ImportError:
    OMGIFOL_AVAILABLE = False
    WAD = None


class WADWriter:
    """High-level wrapper for WAD file creation."""

    def __init__(self, wad_type: str = "PWAD"):
        """Create a new WAD container."""
        if not OMGIFOL_AVAILABLE:
            raise ImportError(
                "omgifol is required for WAD creation. "
                "Install with: pip install omgifol"
            )

        self.wad = WAD()
        self.wad_type = wad_type
        self.maps: dict[str, Any] = {}

    def create_map(self, name: str = "MAP01") -> 'DoomMapData':
        """Create and register a new map."""
        if name in self.maps:
            raise ValueError(f"Map {name} already exists")

        map_data = DoomMapData(name)
        self.maps[name] = map_data
        return map_data

    def get_map(self, name: str) -> 'DoomMapData':
        """Return a previously created map."""
        if name not in self.maps:
            raise KeyError(f"Map {name} does not exist")
        return self.maps[name]

    def add_lump(self, name: str, data: bytes) -> None:
        """Add a non-map lump to the WAD."""
        from omg.lump import Lump
        self.wad.data[name] = Lump(data)

    def save(self, filepath: str | Path) -> None:
        """Write the WAD to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        for name, map_data in self.maps.items():
            self._build_map(name, map_data)

        self.wad.to_file(str(filepath))

    def _build_map(self, name: str, map_data: 'DoomMapData') -> None:
        """Convert `DoomMapData` into omgifol structures."""
        editor = OMGMapEditor()
        use_hexen_format = map_data.behavior is not None

        if use_hexen_format:
            editor.Linedef = ZLinedef
            editor.Thing = ZThing

        for v in map_data.vertices:
            editor.vertexes.append(Vertex(x=v['x'], y=v['y']))

        for s in map_data.sectors:
            sector_kwargs = {
                'z_floor': s.get('floor_height', 0),
                'z_ceil': s.get('ceiling_height', 128),
                'tx_floor': s.get('floor_texture', 'FLOOR4_8'),
                'tx_ceil': s.get('ceiling_texture', 'CEIL3_5'),
                'light': s.get('light_level', 160),
                'type': s.get('special', 0),
                'tag': s.get('tag', 0)
            }
            standard_keys = {'floor_height', 'ceiling_height', 'floor_texture', 'ceiling_texture', 'light_level', 'special', 'tag'}
            for k, v in s.items():
                if k not in standard_keys:
                    sector_kwargs[k] = v

            editor.sectors.append(Sector(**sector_kwargs))

        for sd in map_data.sidedefs:
            side_kwargs = {
                'off_x': sd.get('x_offset', 0),
                'off_y': sd.get('y_offset', 0),
                'tx_up': sd.get('upper_texture', '-') if sd.get('upper_texture') != '-' else '-',
                'tx_low': sd.get('lower_texture', '-') if sd.get('lower_texture') != '-' else '-',
                'tx_mid': sd.get('middle_texture', 'STARTAN2') if sd.get('middle_texture') != '-' else '-',
                'sector': sd.get('sector', 0)
            }
            standard_keys = {'x_offset', 'y_offset', 'upper_texture', 'lower_texture', 'middle_texture', 'sector'}
            for k, v in sd.items():
                if k not in standard_keys:
                    side_kwargs[k] = v

            editor.sidedefs.append(Sidedef(**side_kwargs))

        if use_hexen_format:
            for ld in map_data.linedefs:
                editor.linedefs.append(ZLinedef(
                    vx_a=ld['start_vertex'],
                    vx_b=ld['end_vertex'],
                    flags=ld.get('flags', 1),
                    action=ld.get('special', 0),
                    arg0=ld.get('arg0', 0),
                    arg1=ld.get('arg1', 0),
                    arg2=ld.get('arg2', 0),
                    arg3=ld.get('arg3', 0),
                    arg4=ld.get('arg4', 0),
                    front=ld.get('front_sidedef', -1),
                    back=ld.get('back_sidedef', -1)
                ))
        else:
            for ld in map_data.linedefs:
                editor.linedefs.append(Linedef(
                    vx_a=ld['start_vertex'],
                    vx_b=ld['end_vertex'],
                    flags=ld.get('flags', 1),
                    action=ld.get('special', 0),
                    tag=ld.get('tag', 0),
                    front=ld.get('front_sidedef', -1),
                    back=ld.get('back_sidedef', -1)
                ))

        if use_hexen_format:
            HEXEN_DEFAULT_FLAGS = 7 | 256 | 512 | 1024
            for t in map_data.things:
                thing = ZThing()
                thing.x = t['x']
                thing.y = t['y']
                thing.type = t['type']
                thing.angle = t.get('angle', 0)
                f = t.get('flags', HEXEN_DEFAULT_FLAGS)
                # Doom-format thing flags omit the Hexen game-mode bits, so add
                # them when callers pass a low-value Doom-style flag mask.
                if f < 256:
                    f |= (256 | 512 | 1024)
                thing.flags = f

                thing.tid = t.get('tid', 0)
                thing.height = t.get('height', 0)
                thing.action = t.get('action', 0)
                thing.arg0 = t.get('arg0', 0)
                thing.arg1 = t.get('arg1', 0)
                thing.arg2 = t.get('arg2', 0)
                thing.arg3 = t.get('arg3', 0)
                thing.arg4 = t.get('arg4', 0)
                editor.things.append(thing)
        else:
            for t in map_data.things:
                editor.things.append(Thing(
                    x=t['x'],
                    y=t['y'],
                    angle=t.get('angle', 0),
                    type=t['type'],
                    flags=t.get('flags', 7)
                ))

        # omgifol leaves node-related lumps to an external node builder.
        map_lumps = editor.to_lumps()

        if map_data.behavior:
            from omg.lump import Lump
            map_lumps['BEHAVIOR'] = Lump(map_data.behavior)

        if map_data.scripts:
            from omg.lump import Lump
            scripts_data = map_data.scripts.encode('utf-8') + b'\x00'
            map_lumps['SCRIPTS'] = Lump(scripts_data)

        self.wad.maps[name] = map_lumps


def run_zdbsp(wad_path: str | Path, zdbsp_path: str = "zdbsp") -> bool:
    """Run ZDBSP in place to build node-related lumps for a WAD."""
    wad_path = Path(wad_path)
    resolved = shutil.which(zdbsp_path)
    if resolved is None:
        return False

    tmp_path = wad_path.with_suffix(".zdbsp_tmp.wad")
    try:
        subprocess.run(
            [resolved, "-o", tmp_path, wad_path],
            capture_output=True,
            text=True,
            check=True,
        )
        tmp_path.replace(wad_path)
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class DoomMapData:
    """Intermediate geometry and entity data before WAD compilation."""

    def __init__(self, name: str):
        """Create an empty map data container."""
        self.name = name
        self.vertices: list[dict] = []
        self.linedefs: list[dict] = []
        self.sidedefs: list[dict] = []
        self.sectors: list[dict] = []
        self.things: list[dict] = []
        self.behavior: Optional[bytes] = None
        self.scripts: Optional[str] = None

        self._vertex_index: dict[tuple[int, int], int] = {}

    def add_vertex(self, x: int, y: int) -> int:
        """Add a vertex, reusing an existing one if present."""
        x, y = int(round(x)), int(round(y))
        key = (x, y)

        if key in self._vertex_index:
            return self._vertex_index[key]

        index = len(self.vertices)
        self.vertices.append({'x': x, 'y': y})
        self._vertex_index[key] = index
        return index

    def add_sector(
        self,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_texture: str = 'FLOOR4_8',
        ceiling_texture: str = 'CEIL3_5',
        light_level: int = 160,
        special: int = 0,
        tag: int = 0,
        **kwargs
    ) -> int:
        """Add a sector and return its index."""
        index = len(self.sectors)
        sector_data = {
            'floor_height': floor_height,
            'ceiling_height': ceiling_height,
            'floor_texture': floor_texture,
            'ceiling_texture': ceiling_texture,
            'light_level': light_level,
            'special': special,
            'tag': tag
        }
        sector_data.update(kwargs)
        self.sectors.append(sector_data)
        return index

    def add_sidedef(
        self,
        sector: int,
        upper_texture: str = '-',
        middle_texture: str = 'STARTAN2',
        lower_texture: str = '-',
        x_offset: int = 0,
        y_offset: int = 0,
        **kwargs
    ) -> int:
        """Add a sidedef and return its index."""
        index = len(self.sidedefs)
        side_data = {
            'sector': sector,
            'upper_texture': upper_texture,
            'middle_texture': middle_texture,
            'lower_texture': lower_texture,
            'x_offset': x_offset,
            'y_offset': y_offset
        }
        side_data.update(kwargs)
        self.sidedefs.append(side_data)
        return index

    def add_linedef(
        self,
        start_vertex: int,
        end_vertex: int,
        front_sidedef: int,
        back_sidedef: int = -1,
        flags: int = 1,
        special: int = 0,
        tag: int = 0,
        args: Optional[list[int]] = None
    ) -> int:
        """Add a linedef and return its index."""
        index = len(self.linedefs)
        ld = {
            'start_vertex': start_vertex,
            'end_vertex': end_vertex,
            'front_sidedef': front_sidedef,
            'back_sidedef': back_sidedef,
            'flags': flags,
            'special': special,
            'tag': tag
        }
        if args:
            for i, arg in enumerate(args):
                if i < 5:
                    ld[f'arg{i}'] = arg

        self.linedefs.append(ld)
        return index

    def add_thing(
        self,
        x: int,
        y: int,
        thing_type: int,
        angle: int = 0,
        flags: int = 7,
        tid: int = 0,
        **kwargs
    ) -> int:
        """Add a thing and return its index."""
        index = len(self.things)
        thing_data = {
            'x': int(round(x)),
            'y': int(round(y)),
            'type': thing_type,
            'angle': angle,
            'flags': flags,
            'tid': tid
        }

        if 'args' in kwargs:
            args_list = kwargs.pop('args')
            for i, arg in enumerate(args_list):
                if i < 5:
                    kwargs[f'arg{i}'] = arg

        thing_data.update(kwargs)
        self.things.append(thing_data)
        return index
