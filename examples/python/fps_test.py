#!/usr/bin/env python3

#####################################################################
# This script tests performance in frames per second.
# Change iters, resolution, window visibility, use get_ state or not.
# It should give you some idea of how fast the framework can work on
# your hardware. The test involves copying the state to make it more
# similar to any reasonable usage. Comment the line with get_state
# to exclude copying process.
#####################################################################

import os
from argparse import ArgumentParser
from random import choice
from time import time

import tqdm

import vizdoom as vzd


# Rendering options:
resolution = vzd.ScreenResolution.RES_320X240
screen_format = vzd.ScreenFormat.CRCGCB
depth_buffer = False
labels_buffer = False
automap_buffer = False
audio_buffer = False
objects_info = False
sectors_info = False

#####################################################################
DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "basic.cfg")
DEFAULT_ITERATIONS = 10000

if __name__ == "__main__":

    parser = ArgumentParser("ViZDoom example showing possible framerates.")
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=DEFAULT_ITERATIONS,
        type=int,
        help="Number of iterations(actions) to run",
    )
    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)

    game.set_screen_resolution(resolution)
    game.set_screen_format(screen_format)

    game.set_depth_buffer_enabled(depth_buffer)
    game.set_labels_buffer_enabled(labels_buffer)
    game.set_automap_buffer_enabled(automap_buffer)
    game.set_audio_buffer_enabled(audio_buffer)
    game.set_objects_info_enabled(objects_info)
    game.set_sectors_info_enabled(sectors_info)

    game.set_window_visible(False)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    left = actions[0]
    right = actions[1]
    shoot = actions[2]
    idle = [False, False, False]

    start = time()

    print("Checking FPS with selected features. It may take some time. Be patient.")

    for i in tqdm.trange(args.iterations, leave=False):

        if game.is_episode_finished():
            game.new_episode()

        # Copying happens here
        s = game.get_state()
        game.make_action(choice(actions))

    end = time()
    t = end - start
    print("Results:")
    print("Iterations:", args.iterations)
    print("Resolution:", resolution)
    print("time:", round(t, 3), "s")
    print("fps: ", round(args.iterations / t, 2))

    game.close()
