#!/usr/bin/env python3

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair a macOS wheel and bundle OpenAL Soft and SDL3."
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--require-archs", required=True)
    parser.add_argument("--sdl3-library", type=Path)
    return parser.parse_args()


def wheel_deployment_target(wheel):
    versions = re.findall(r"macosx_(\d+)_(\d+)_", wheel.name)
    if not versions:
        raise RuntimeError(f"No macOS deployment target in wheel name: {wheel.name}")
    return ".".join(str(part) for part in min(tuple(map(int, v)) for v in versions))


def required_architectures(require_archs):
    return {"universal2": ["x86_64", "arm64"], "intel": ["i386", "x86_64"]}.get(
        require_archs, require_archs.split(",")
    )


def homebrew_library(formula, filename):
    prefix = Path(
        subprocess.check_output(["brew", "--prefix", formula], text=True).strip()
    )
    library = prefix / "lib" / filename
    if not library.is_file():
        raise RuntimeError(f"Library not found at {library}")
    return library


def bundle_files(wheel, destination, files):
    bundled_files = {name: path.read_bytes() for name, path in files.items()}

    with zipfile.ZipFile(wheel, "r") as source:
        record_infos = [
            info
            for info in source.infolist()
            if info.filename.endswith(".dist-info/RECORD")
        ]
        if len(record_infos) != 1:
            raise RuntimeError(
                f"Expected one .dist-info/RECORD in {wheel}, found {record_infos}"
            )
        record_info = record_infos[0]
        record_rows = list(
            csv.reader(io.StringIO(source.read(record_info).decode("utf-8")))
        )
        record_rows = [row for row in record_rows if row[0] not in bundled_files]
        record_indexes = [
            index
            for index, row in enumerate(record_rows)
            if row[0] == record_info.filename
        ]
        if len(record_indexes) != 1:
            raise RuntimeError(
                f"Expected one entry for {record_info.filename} in RECORD, "
                f"found {record_indexes}"
            )
        for name, data in bundled_files.items():
            bundled_hash = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            record_rows.insert(
                record_indexes[0],
                [
                    name,
                    f"sha256={bundled_hash.rstrip(b'=').decode('ascii')}",
                    str(len(data)),
                ],
            )

        record_output = io.StringIO(newline="")
        csv.writer(record_output, lineterminator="\n").writerows(record_rows)

        with zipfile.ZipFile(destination, "w", allowZip64=True) as target:
            for info in source.infolist():
                if (
                    info.filename not in bundled_files
                    and info.filename != record_info.filename
                ):
                    target.writestr(info, source.read(info))

            for name, data in bundled_files.items():
                file_info = zipfile.ZipInfo(name, record_info.date_time)
                file_info.create_system = 3
                file_info.external_attr = (
                    0o100755 if name.endswith(".dylib") else 0o100644
                ) << 16
                file_info.compress_type = zipfile.ZIP_DEFLATED
                target.writestr(file_info, data)
            target.writestr(record_info, record_output.getvalue().encode("utf-8"))


def main():
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    if args.sdl3_library is None:
        sdl3_library = homebrew_library("sdl3", "libSDL3.dylib")
    else:
        sdl3_library = args.sdl3_library
    if not sdl3_library.is_file():
        raise RuntimeError(f"SDL3 library not found at {sdl3_library}")
    openal_library = homebrew_library("openal-soft", "libopenal.1.dylib")
    openal_prefix = openal_library.parent.parent
    notices = {
        f"vizdoom/.dylibs/openal-soft/{name}": openal_prefix / name
        for name in ("COPYING", "LICENSE-pffft")
    }

    with tempfile.TemporaryDirectory(prefix="vizdoom-wheel-repair-") as temp_dir:
        temp_path = Path(temp_dir)
        deployment_target = wheel_deployment_target(args.wheel)
        repair_env = os.environ.copy()
        repair_env["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
        repaired_dir = temp_path / "repaired"
        repaired_dir.mkdir()
        subprocess.run(
            [
                "delocate-wheel",
                "--require-archs",
                args.require_archs,
                "-w",
                str(repaired_dir),
                "-v",
                str(args.wheel),
            ],
            check=True,
            env=repair_env,
        )
        (repaired_wheel,) = repaired_dir.glob("*.whl")
        bundled_sdl3 = temp_path / "libSDL3.dylib"
        shutil.copy2(sdl3_library.resolve(), bundled_sdl3)
        bundled_openal = temp_path / "libopenal.1.dylib"
        shutil.copy2(openal_library.resolve(), bundled_openal)

        # These dlopen() dependencies need explicit bundling and architecture
        # checks. A second delocate pass checks their deployment targets and
        # relocates any additional dependencies.
        libraries = [bundled_sdl3, bundled_openal]
        for library in libraries:
            subprocess.run(
                [
                    "lipo",
                    str(library),
                    "-verify_arch",
                    *required_architectures(args.require_archs),
                ],
                check=True,
            )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(library)], check=True
            )
        bundled_wheel = temp_path / repaired_wheel.name
        bundle_files(
            repaired_wheel,
            bundled_wheel,
            {f"vizdoom/.dylibs/{lib.name}": lib for lib in libraries} | notices,
        )
        subprocess.run(
            [
                "delocate-wheel",
                "--require-archs",
                args.require_archs,
                # delocate refuses to copy into an existing bundle directory.
                "--lib-sdir",
                ".dylibs-extra",
                "-w",
                str(args.dest_dir),
                "-v",
                str(bundled_wheel),
            ],
            check=True,
            env=repair_env,
        )


if __name__ == "__main__":
    main()
