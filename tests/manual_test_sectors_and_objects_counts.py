#!/usr/bin/env python3

# This test should be run manually.
# It reports number of objects and sectors for MAP01-MAP32.

import os
import sys
from argparse import ArgumentParser

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "freedoom2.cfg")
MAP_NAMES = [f"map{i:02d}" for i in range(1, 33)]


def count_objects_and_sectors(config_path, map_name):
    game = vzd.DoomGame()
    try:
        game.load_config(config_path)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_doom_map(map_name)
        game.init()

        state = game.get_state()
        objects_count = len(state.objects) if state and state.objects is not None else 0
        sectors_count = len(state.sectors) if state and state.sectors is not None else 0
        return objects_count, sectors_count, None
    except Exception as exc:
        return None, None, str(exc).splitlines()[0]
    finally:
        try:
            game.close()
        except Exception:
            pass


def main():
    parser = ArgumentParser(
        "Manual test that reports object and sector counts for MAP01-MAP32."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help="Path to .cfg file. Defaults to scenarios/doom2.cfg.",
    )
    args = parser.parse_args()

    print(f"Config: {args.config}")
    print(f"{'Map':<8}{'Objects':>10}{'Sectors':>10}  Status")
    print("-" * 50)

    rows = []
    for map_name in MAP_NAMES:
        objects_count, sectors_count, error = count_objects_and_sectors(
            args.config, map_name
        )
        if error is None:
            status = "OK"
            print(
                f"{map_name.upper():<8}{objects_count:>10}{sectors_count:>10}  {status}"
            )
            rows.append((map_name, objects_count, sectors_count))
        else:
            status = f"ERROR: {error}"
            print(f"{map_name.upper():<8}{'-':>10}{'-':>10}  {status}")

    print("-" * 50)
    if rows:
        max_objects = max(rows, key=lambda row: row[1])
        max_sectors = max(rows, key=lambda row: row[2])
        print(
            f"Max objects: {max_objects[1]} on {max_objects[0].upper()}, "
            f"max sectors: {max_sectors[2]} on {max_sectors[0].upper()}."
        )
    else:
        print("No maps initialized successfully.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
