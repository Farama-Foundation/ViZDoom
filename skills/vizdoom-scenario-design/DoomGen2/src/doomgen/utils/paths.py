import os
import platform
import shutil
import struct
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    # Traverse upward to find project root
    # The structure is: DoomGen/src/doomgen/utils/paths.py
    # So we need to go up 3 levels to reach DoomGen/
    while current != current.parent:
        # Check if this is the project root (has 'src' directory)
        if (current / "src").is_dir() and (current / "pyproject.toml").is_file():
            return current
        current = current.parent
    raise RuntimeError("Cannot determine project root. Expected to find 'src' directory and 'pyproject.toml' at the project root.")


def get_bundled_compiler(tool_name: str) -> str:
    if tool_name not in ["acc", "zdbsp"]:
        raise ValueError(f"Unknown tool: {tool_name}")

    project_root = get_project_root()
    is_windows = sys.platform == "win32" or sys.platform == "cygwin"
    exe_ext = ".exe" if is_windows else ""

    tool_path = project_root / tool_name / f"{tool_name}{exe_ext}"
    return str(tool_path.resolve())


def binary_matches_platform(path: str | Path) -> bool:
    """Return True if ``path`` appears executable on the current OS/architecture."""
    file_path = Path(path)
    if not file_path.is_file():
        return False
    try:
        with file_path.open("rb") as fh:
            header = fh.read(8)
    except OSError:
        return False
    if len(header) < 4:
        return False
    if sys.platform in ("win32", "cygwin"):
        return header[:2] == b"MZ"
    if header[:2] == b"#!":
        return True
    if sys.platform.startswith("linux"):
        return header[:4] == b"\x7fELF"
    if sys.platform == "darwin":
        magic = header[:4]
        if magic in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
            return True  # universal Mach-O
        if magic == b"\xcf\xfa\xed\xfe" and len(header) >= 8:
            cputype = struct.unpack("<I", header[4:8])[0]
            machine = platform.machine().lower()
            if cputype == 0x0100000C:  # CPU_TYPE_ARM64
                return machine in ("arm64", "aarch64")
            if cputype == 0x01000007:  # CPU_TYPE_X86_64
                return machine in ("x86_64", "amd64")
        return magic in (b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce")
    return True


def resolve_compiler(tool_name: str, override: str | None = None) -> str | None:
    """Resolve ``acc`` or ``zdbsp`` from override, compatible bundled binary, or PATH."""
    if override is not None:
        return shutil.which(override) or (
            str(Path(override).resolve()) if Path(override).is_file() else None
        )
    bundled = Path(get_bundled_compiler(tool_name))
    if bundled.is_file() and binary_matches_platform(bundled):
        return str(bundled)
    return shutil.which(tool_name)


def validate_executable(executable_path: str) -> bool:
    resolved = shutil.which(executable_path)
    if resolved is not None:
        return True
    executable_path_obj = Path(executable_path)
    return executable_path_obj.exists() and executable_path_obj.is_file()
