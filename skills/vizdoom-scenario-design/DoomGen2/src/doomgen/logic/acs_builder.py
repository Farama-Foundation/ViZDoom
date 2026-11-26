"""Pythonic ACS (Action Code Script) generation."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from doomgen.utils.paths import get_bundled_compiler, validate_executable


class ScriptType(Enum):
    """Types of ACS scripts."""
    OPEN = auto()       # Runs when map opens
    ENTER = auto()      # Runs when player enters
    RESPAWN = auto()    # Runs when player respawns
    DEATH = auto()      # Runs when player dies
    LIGHTNING = auto()  # Runs periodically for lightning
    UNLOADING = auto()  # Runs when map unloads
    DISCONNECT = auto() # Runs when player disconnects
    RETURN = auto()     # Runs when returning to map
    EVENT = auto()      # Custom event script
    KILL = auto()       # Runs when thing is killed
    REOPEN = auto()     # Runs when returning to hub
    VOID = auto()       # Callable script (void)


class VariableScope(Enum):
    """Scopes for ACS variables."""
    MAP = auto()        # Visible to all scripts in the map
    WORLD = auto()      # Visible across hub (reset on hub entry)
    GLOBAL = auto()     # Visible across all maps (persistent)


@dataclass
class ACSVariable:
    """An ACS variable declaration."""
    name: str
    var_type: str
    scope: VariableScope
    initial: Any = None
    index: int | None = None

    def to_code(self) -> str:
        """Render the ACS declaration."""
        if self.scope == VariableScope.MAP:
            line = f"{self.var_type} {self.name}"
            if self.initial is not None:
                line += f" = {self.initial}"
            return line + ";"

        elif self.scope == VariableScope.WORLD:
            if self.index is None:
                raise ValueError(f"World variable '{self.name}' must have an index.")
            return f"world {self.var_type} {self.index}:{self.name};"

        elif self.scope == VariableScope.GLOBAL:
            if self.index is None:
                raise ValueError(f"Global variable '{self.name}' must have an index.")
            return f"global {self.var_type} {self.index}:{self.name};"

        return ""


@dataclass
class ACSScript:
    """A single ACS script body with its trigger metadata."""
    number: int
    script_type: ScriptType
    body: str
    args: list[str] = field(default_factory=list)

    def to_code(self) -> str:
        """Generate ACS code for this script."""
        type_str = {
            ScriptType.OPEN: "OPEN",
            ScriptType.ENTER: "ENTER",
            ScriptType.RESPAWN: "RESPAWN",
            ScriptType.DEATH: "DEATH",
            ScriptType.LIGHTNING: "LIGHTNING",
            ScriptType.UNLOADING: "UNLOADING",
            ScriptType.DISCONNECT: "DISCONNECT",
            ScriptType.RETURN: "RETURN",
            ScriptType.EVENT: "EVENT",
            ScriptType.KILL: "KILL",
            ScriptType.REOPEN: "REOPEN",
            ScriptType.VOID: "",
        }.get(self.script_type, "")

        # Build argument list
        if self.args:
            args_str = ", ".join(f"int {arg}" for arg in self.args)
            header = f"script {self.number} ({args_str})"
        elif type_str:
            header = f"script {self.number} {type_str}"
        else:
            header = f"script {self.number} (void)"

        return f"{header}\n{{\n{self._indent(self.body)}\n}}"

    def _indent(self, code: str, level: int = 1) -> str:
        """Indent code lines."""
        indent = "    " * level
        lines = code.strip().split("\n")
        return "\n".join(indent + line for line in lines)


class ACSBuilder:
    """Builds ACS source and compiles it with ACC."""

    def __init__(self):
        """Create an empty ACS builder."""
        self.scripts: list[ACSScript] = []
        self.includes: list[str] = []
        self.defines: dict[str, Any] = {}
        self.variables: list[ACSVariable] = []
        self.global_code_blocks: list[str] = []
        self._next_script_num = 1

    def add_global_code(self, code: str) -> ACSBuilder:
        """Add a raw top-level ACS block."""
        self.global_code_blocks.append(code)
        return self

    def add_include(self, filename: str) -> ACSBuilder:
        """Add an ACS include."""
        self.includes.append(filename)
        return self

    def add_define(self, name: str, value: Any) -> ACSBuilder:
        """Add a `#define` macro."""
        self.defines[name] = value
        return self

    def add_map_var(
        self,
        name: str,
        var_type: str = "int",
        initial: Any = None
    ) -> ACSBuilder:
        """Add a map-scope ACS variable."""
        self.variables.append(ACSVariable(name, var_type, VariableScope.MAP, initial=initial))
        return self

    def add_world_var(
        self,
        name: str,
        index: int,
        var_type: str = "int"
    ) -> ACSBuilder:
        """Add a world-scope ACS variable."""
        self.variables.append(ACSVariable(name, var_type, VariableScope.WORLD, index=index))
        return self

    def add_global_var(
        self,
        name: str,
        index: int,
        var_type: str = "int"
    ) -> ACSBuilder:
        """Add a global-scope ACS variable."""
        self.variables.append(ACSVariable(name, var_type, VariableScope.GLOBAL, index=index))
        return self

    def add_script(
        self,
        script_type: ScriptType,
        body: str,
        number: int | None = None,
        args: list[str] | None = None
    ) -> int:
        """Add a script and return its script number."""
        if number is None:
            number = self._next_script_num
            self._next_script_num += 1
        else:
            self._next_script_num = max(self._next_script_num, number + 1)

        script = ACSScript(
            number=number,
            script_type=script_type,
            body=body,
            args=args or []
        )

        self.scripts.append(script)
        return number

    def define_door_logic(
        self,
        tag: int,
        open_time: int = 35,  # Tics (35 = 1 second)
        close_time: int = 105,  # Tics until auto-close
        speed: int = 16
    ) -> int:
        """Create a door open/close script."""
        body = f"""
Door_Open({tag}, {speed});
Delay({open_time});
Door_Close({tag}, {speed});
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def define_teleport_logic(
        self,
        source_tag: int,
        dest_tag: int
    ) -> int:
        """Create a teleporter script."""
        body = f"""
Teleport({dest_tag}, 0, 0);
"""
        return self.add_script(ScriptType.ENTER, body.strip())

    def define_spawn_enemies(
        self,
        thing_type: int,
        spawn_tag: int,
        count: int = 1,
        delay: int = 35
    ) -> int:
        """Create a repeated enemy spawn script."""
        body = f"""
for (int i = 0; i < {count}; i++)
{{
    SpawnSpot("{thing_type}", {spawn_tag});
    Delay({delay});
}}
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def define_message(
        self,
        message: str,
        script_type: ScriptType = ScriptType.ENTER
    ) -> int:
        """Create a message display script."""
        escaped = message.replace('"', '\\"')
        body = f'Print(s:"{escaped}");'
        return self.add_script(script_type, body)

    def define_sector_light(
        self,
        tag: int,
        min_light: int = 64,
        max_light: int = 255,
        pulse_time: int = 35
    ) -> int:
        """Create a pulsing light effect."""
        body = f"""
while (true)
{{
    Light_Fade({tag}, {max_light}, {pulse_time});
    Delay({pulse_time});
    Light_Fade({tag}, {min_light}, {pulse_time});
    Delay({pulse_time});
}}
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def to_code(self) -> str:
        """Render the complete ACS source file."""
        lines = []

        for inc in self.includes:
            lines.append(f'#include "{inc}"')

        if self.includes:
            lines.append("")

        for name, value in self.defines.items():
            lines.append(f"#define {name} {value}")

        if self.defines:
            lines.append("")

        for var in self.variables:
            lines.append(var.to_code())

        if self.variables:
            lines.append("")

        for block in self.global_code_blocks:
            lines.append(block)
            lines.append("")

        for script in self.scripts:
            lines.append(script.to_code())
            lines.append("")

        return "\n".join(lines)

    def compile(
        self,
        output_path: str | Path | None = None,
        acc_path: str | None = None
    ) -> bytes | None:
        """Compile the scripts with ACC and return the bytecode."""
        if acc_path is None:
            acc_path = get_bundled_compiler("acc")

        source_code = self.to_code()

        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "script.acs"
            source_file.write_text(source_code)

            if output_path:
                obj_file = Path(output_path)
            else:
                obj_file = Path(tmpdir) / "script.o"

            if not validate_executable(acc_path):
                raise FileNotFoundError(
                    f"ACC compiler not found at '{acc_path}'. "
                    "Ensure ACC is installed or available via bundled ./acc directory."
                )

            try:
                subprocess.run(
                    [acc_path, str(source_file), str(obj_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"ACC compiler not found at '{acc_path}'. "
                    "Please install ACC and add it to your PATH."
                )

            if obj_file.exists():
                return obj_file.read_bytes()

            return None

    def save_source(self, filepath: str | Path) -> None:
        """Write the rendered ACS source to disk."""
        Path(filepath).write_text(self.to_code())
