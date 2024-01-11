#!/usr/bin/env python3

# Tests ViZDoom load_config method and all the config files from the scenario directory.
# This test can be run as Python script or via PyTest

import os

import vizdoom as vzd


def test_load_config():
    print("Testing load_config() and default scenarios ...")

    for file in os.listdir(vzd.scenarios_path):
        if file.endswith(".cfg"):
            game = vzd.DoomGame()

            # Both should work
            game.load_config(os.path.join(vzd.scenarios_path, file))
            game.load_config(file)

            w = game.get_screen_width()
            h = game.get_screen_height()
            assert (
                w == 320 and h == 240
            ), f"Config file {file} is using non-default screen resolution: {w}x{h} instead 320x240."


if __name__ == "__main__":
    test_load_config()
