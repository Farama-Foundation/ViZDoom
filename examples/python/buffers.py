#!/usr/bin/env python3

#####################################################################
# This script presents different buffers and formats.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from random import choice
import vizdoom as vzd
from argparse import ArgumentParser

DEFAULT_CONFIG = "../../scenarios/deadly_corridor.cfg"

import cv2

if __name__ == "__main__":

    parser = ArgumentParser("ViZDoom example showing different buffers (screen, depth, labels).")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)

    # game.set_console_enabled(True)
    # game.set_window_visible(False)

    # Just umcomment desired format for screen buffer (and map buffer).
    # The last uncommented will be applied.
    # Formats with C (CRCGCB and CBCGCR) were omitted cause they are not cv2 friendly.
    # Default format is ScreenFormat.CRCGCB.

    # OpenCV uses a BGR colorspace by default.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # game.set_screen_format(ScreenFormat.RGB24)
    # game.set_screen_format(ScreenFormat.RGBA32)
    # game.set_screen_format(ScreenFormat.ARGB32)
    # game.set_screen_format(ScreenFormat.BGRA32)
    # game.set_screen_format(ScreenFormat.ABGR32)
    # game.set_screen_format(ScreenFormat.GRAY8)

    # Raw Doom buffer with palette's values. This one makes no sense in particular
    # game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

    # Sets resolution for all buffers.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_render_hud(True)
    game.set_render_minimal_hud(False)

    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10
    sleep_time = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():
            # Gets the state and possibly do something with it
            state = game.get_state()

            # Display all the buffers here!

            # Just screen buffer, given in selected format. This buffer is always available.
            screen = state.screen_buffer
            cv2.imshow('ViZDoom Screen Buffer', screen)

            # Depth buffer, always in 8-bit gray channel format.
            # This is most fun. It looks best if you inverse colors.
            depth = state.depth_buffer
            if depth is not None:
                cv2.imshow('ViZDoom Depth Buffer', depth)

            # Labels buffer, always in 8-bit gray channel format.
            # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
            # Labels data are available in state.labels, also see labels.py example.
            labels = state.labels_buffer
            if labels is not None:
                cv2.imshow('ViZDoom Labels Buffer', labels)

            # Map buffer, in the same format as screen buffer.
            # Shows top down map of the current episode/level.
            automap = state.automap_buffer
            if automap is not None:
                cv2.imshow('ViZDoom Map Buffer', automap)

            cv2.waitKey(int(sleep_time * 1000))

            game.make_action(choice(actions))

            print("State #" + str(state.number))
            print("=====================")

        print("Episode finished!")
        print("************************")

    cv2.destroyAllWindows()