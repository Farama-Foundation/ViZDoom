import base64
import csv
import hashlib
import importlib.util
import io
import stat
import zipfile
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_macos_wheel.py"
SPEC = importlib.util.spec_from_file_location("repair_macos_wheel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


@pytest.mark.parametrize("replace_existing", [False, True])
def test_bundle_files_updates_wheel_record(tmp_path, replace_existing):
    wheel = tmp_path / "original.whl"
    destination = tmp_path / "bundled.whl"
    record_name = "vizdoom-1.3.2.dist-info/RECORD"
    sdl_name = "vizdoom/.dylibs/libSDL3.dylib"
    records = [["vizdoom/__init__.py", "", ""], [record_name, "", ""]]
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vizdoom/__init__.py", b"original package contents")
        if replace_existing:
            archive.writestr(sdl_name, b"old SDL3")
            records.insert(0, [sdl_name, "sha256=old", "8"])
        record = io.StringIO(newline="")
        csv.writer(record).writerows(records)
        archive.writestr(record_name, record.getvalue())

    files = {}
    for name, data in {
        sdl_name: b"new SDL3",
        "vizdoom/.dylibs/libopenal.1.dylib": b"OpenAL Soft",
        "vizdoom/.dylibs/openal-soft/COPYING": b"OpenAL Soft license",
        "vizdoom/.dylibs/openal-soft/source.tar.bz2": b"OpenAL Soft source archive",
    }.items():
        path = tmp_path / Path(name).name
        path.write_bytes(data)
        files[name] = path

    repair.bundle_files(wheel, destination, files)

    with zipfile.ZipFile(destination) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert archive.read("vizdoom/__init__.py") == b"original package contents"
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        assert len(rows) == len({row[0] for row in rows})
        entries = {row[0]: row[1:] for row in rows}
        assert entries[record_name] == ["", ""]
        for name, path in files.items():
            data = path.read_bytes()
            assert archive.read(name) == data
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                b"="
            )
            assert entries[name] == [f"sha256={digest.decode()}", str(len(data))]
            permissions = stat.S_IMODE(archive.getinfo(name).external_attr >> 16)
            assert permissions == (0o755 if name.endswith(".dylib") else 0o644)


@pytest.mark.parametrize(
    "wheel_name,target",
    [
        ("vizdoom-1.3.2-cp314-cp314-macosx_14_0_arm64.whl", "14.0"),
        ("vizdoom-1.3.2-cp310-cp310-macosx_12_0_x86_64.whl", "12.0"),
        (
            "vizdoom-1.3.2-cp310-cp310-macosx_14_0_x86_64.macosx_11_0_arm64.whl",
            "11.0",
        ),
    ],
)
def test_wheel_deployment_target(wheel_name, target):
    assert repair.wheel_deployment_target(Path(wheel_name)) == target


def test_wheel_deployment_target_requires_macos():
    with pytest.raises(RuntimeError, match="No macOS deployment target"):
        repair.wheel_deployment_target(Path("vizdoom-1.3.2-py3-none-any.whl"))


@pytest.mark.parametrize("cached", [False, True])
def test_openal_source_checksum_is_verified(tmp_path, monkeypatch, cached):
    if cached:
        (tmp_path / f"openal-soft-{repair.OPENAL_VERSION}.tar.bz2").write_bytes(
            b"incorrect cached archive"
        )
    monkeypatch.setattr(
        repair.urllib.request, "urlopen", lambda url: io.BytesIO(b"incorrect archive")
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        repair.build_openal_soft(tmp_path, "arm64", "14.0")
