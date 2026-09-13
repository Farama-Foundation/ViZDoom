#!/usr/bin/env python3

import argparse
import base64
import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair a macOS wheel and bundle SDL3 for sdl2-compat."
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--require-archs", required=True)
    parser.add_argument("--sdl3-library", type=Path)
    return parser.parse_args()


def find_single_path(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one path matching {pattern!r} in {directory}, found {paths}"
        )
    return paths[0]


def bundle_library(wheel, destination, library):
    bundled_path = "vizdoom/.dylibs/libSDL3.dylib"
    bundled_data = library.read_bytes()
    bundled_hash = base64.urlsafe_b64encode(
        hashlib.sha256(bundled_data).digest()
    ).rstrip(b"=")

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
        record_rows = [row for row in record_rows if row[0] != bundled_path]
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
        record_rows.insert(
            record_indexes[0],
            [
                bundled_path,
                f"sha256={bundled_hash.decode('ascii')}",
                str(len(bundled_data)),
            ],
        )

        record_output = io.StringIO(newline="")
        csv.writer(record_output, lineterminator="\n").writerows(record_rows)

        with zipfile.ZipFile(destination, "w", allowZip64=True) as target:
            for info in source.infolist():
                if info.filename not in (bundled_path, record_info.filename):
                    target.writestr(info, source.read(info))

            library_info = zipfile.ZipInfo(bundled_path, record_info.date_time)
            library_info.create_system = 3
            library_info.external_attr = 0o100755 << 16
            library_info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(library_info, bundled_data)
            target.writestr(record_info, record_output.getvalue().encode("utf-8"))


def main():
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    if args.sdl3_library is None:
        sdl3_prefix = Path(
            subprocess.check_output(["brew", "--prefix", "sdl3"], text=True).strip()
        )
        sdl3_library = sdl3_prefix / "lib" / "libSDL3.dylib"
    else:
        sdl3_library = args.sdl3_library
    if not sdl3_library.is_file():
        raise RuntimeError(f"SDL3 library not found at {sdl3_library}")

    with tempfile.TemporaryDirectory(prefix="vizdoom-wheel-repair-") as temp_dir:
        temp_path = Path(temp_dir)
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
        )

        repaired_wheel = find_single_path(repaired_dir, "*.whl")
        bundled_library = temp_path / "libSDL3.dylib"
        shutil.copy2(sdl3_library.resolve(), bundled_library)

        # Homebrew's sdl2-compat loads this library with dlopen(), so delocate
        # cannot discover or sign it as an ordinary Mach-O dependency.
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(bundled_library)], check=True
        )
        bundle_library(
            repaired_wheel,
            args.dest_dir / repaired_wheel.name,
            bundled_library,
        )


if __name__ == "__main__":
    main()
