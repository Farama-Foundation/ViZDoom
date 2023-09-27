#!/usr/bin/env python3

# Tests for ViZDoom enums and related methods.
# This test can be run as Python script or via PyTest

import os
import vizdoom as vzd


def test_load_config():
    for file in os.listdir(vzd.scenarios_path):
        if file.endswith(".cfg"):
            vzd.DoomGame().load_config(os.path.join(vzd.scenarios_path, file))
            vzd.DoomGame().load_config(file)


if __name__ == "__main__":
    test_load_config()
