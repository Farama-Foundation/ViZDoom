#!/usr/bin/env python3

# This test should be run manually.
# It saves screen/depth/labels for every render combination and automap for every automap combination.

import os
from argparse import ArgumentParser
from itertools import product

import cv2

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "my_way_home.cfg")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "manual_test_buffers_output"
)

RENDER_OPTIONS = {
    "hud": {
        "setter": "set_render_hud",
        "values": [False, True],
    },
    "minimal_hud": {
        "setter": "set_render_minimal_hud",
        "values": [False, True],
    },
    "weapon": {
        "setter": "set_render_weapon",
        "values": [False, True],
    },
    "crosshair": {
        "setter": "set_render_crosshair",
        "values": [False, True],
    },
    "messages": {
        "setter": "set_render_messages",
        "values": [False, True],
    },
}

AUTOMAP_OPTIONS = {
    "automap_mode": {
        "setter": "set_automap_mode",
        "values": [
            vzd.AutomapMode.NORMAL,
            vzd.AutomapMode.WHOLE,
            vzd.AutomapMode.OBJECTS,
            vzd.AutomapMode.OBJECTS_WITH_SIZE,
        ],
        "slug_prefix": "amode",
        "slug_value": lambda value: value.name.lower(),
    },
    "automap_rotate": {
        "setter": "set_automap_rotate",
        "values": [False, True],
        "slug_prefix": "arotate",
    },
    "automap_textures": {
        "setter": "set_automap_render_textures",
        "values": [False, True],
        "slug_prefix": "atex",
    },
    "automap_sprites": {
        "setter": "set_automap_render_objects_as_sprites",
        "values": [False, True],
        "slug_prefix": "asprites",
    },
    "messages": {
        "setter": "set_render_messages",
        "values": [False, True],
    },
}

DEFAULT_RENDER_SETTINGS = {
    name: spec["values"][0] for name, spec in RENDER_OPTIONS.items()
}
DEFAULT_AUTOMAP_SETTINGS = {
    name: spec["values"][0] for name, spec in AUTOMAP_OPTIONS.items()
}


def iter_settings(options):
    names = list(options.keys())
    values_list = [options[name]["values"] for name in names]
    for values in product(*values_list):
        yield dict(zip(names, values))


def settings_slug(options, settings):
    parts = []
    for name, spec in options.items():
        prefix = spec.get("slug_prefix", name)
        value = settings[name]
        slug_value = spec.get("slug_value")
        if slug_value is not None:
            value_str = slug_value(value)
        elif isinstance(value, bool):
            value_str = "1" if value else "0"
        else:
            value_str = str(value)
        parts.append(f"{prefix}={value_str}")
    return "_".join(parts)


def apply_options(game, options, settings):
    for name, spec in options.items():
        getattr(game, spec["setter"])(settings[name])


def count_combos(options):
    total = 1
    for spec in options.values():
        total *= len(spec["values"])
    return total


def setup_game(
    config,
    seed,
    render_settings,
    automap_settings,
    enable_depth=True,
    enable_labels=True,
    enable_automap=True,
):
    game = vzd.DoomGame()
    game.load_config(config)
    if seed is not None:
        game.set_seed(seed)

    game.set_window_visible(False)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    game.set_depth_buffer_enabled(enable_depth)
    game.set_labels_buffer_enabled(enable_labels)
    game.set_automap_buffer_enabled(enable_automap)

    apply_options(game, RENDER_OPTIONS, render_settings)

    if enable_automap:
        apply_options(game, AUTOMAP_OPTIONS, automap_settings)

    game.init()
    return game


def save_buffers(output_dir, slug, state, buffer_names):
    available = {
        "screen": state.screen_buffer,
        "depth": state.depth_buffer,
        "labels": state.labels_buffer,
        "automap": state.automap_buffer,
    }

    for name in buffer_names:
        buffer = available[name]
        if buffer is None:
            print(f"  Skipping {name} buffer (not available)")
            continue

        buffer_dir = os.path.join(output_dir, name)
        os.makedirs(buffer_dir, exist_ok=True)
        path = os.path.join(buffer_dir, f"{slug}.png")
        if not cv2.imwrite(path, buffer):
            raise RuntimeError(f"Failed to write buffer {name} to {path}")


def main():
    parser = ArgumentParser(
        "Manual test that saves buffers for render and automap combinations."
    )
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for saved buffers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="ViZDoom RNG seed for deterministic episodes.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Optional cap on the number of combinations to run per section.",
    )
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    render_combos = count_combos(RENDER_OPTIONS)
    automap_combos = count_combos(AUTOMAP_OPTIONS)

    print(f"Render combinations: {render_combos}")
    print(f"Automap combinations: {automap_combos}")

    # combo_index = 0
    # for render_settings in iter_settings(RENDER_OPTIONS):
    #     combo_index += 1
    #     if args.max_combos is not None and combo_index > args.max_combos:
    #         print("Reached max render combos limit.")
    #         break

    #     slug = settings_slug(RENDER_OPTIONS, render_settings)
    #     print(f"[render {combo_index}/{render_combos}] {slug}")

    #     game = setup_game(
    #         args.config,
    #         args.seed,
    #         render_settings,
    #         DEFAULT_AUTOMAP_SETTINGS,
    #         enable_depth=True,
    #         enable_labels=True,
    #         enable_automap=False,
    #     )
    #     try:
    #         game.new_episode()
    #         game.advance_action(tics=10)
    #         state = game.get_state()
    #         assert state is not None
    #         save_buffers(
    #             args.output_dir,
    #             slug,
    #             state,
    #             buffer_names=["screen", "depth", "labels"],
    #         )
    #     finally:
    #         game.close()

    combo_index = 0
    for automap_settings in iter_settings(AUTOMAP_OPTIONS):
        combo_index += 1
        if args.max_combos is not None and combo_index > args.max_combos:
            print("Reached max automap combos limit.")
            break

        slug = settings_slug(AUTOMAP_OPTIONS, automap_settings)
        print(f"[automap {combo_index}/{automap_combos}] {slug}")

        game = setup_game(
            args.config,
            args.seed,
            DEFAULT_RENDER_SETTINGS,
            automap_settings,
            enable_depth=False,
            enable_labels=False,
            enable_automap=True,
        )
        try:
            game.new_episode()
            game.advance_action(tics=10)
            state = game.get_state()
            assert state is not None
            save_buffers(
                args.output_dir,
                slug,
                state,
                buffer_names=["automap"],
            )
        finally:
            game.close()


if __name__ == "__main__":
    main()
