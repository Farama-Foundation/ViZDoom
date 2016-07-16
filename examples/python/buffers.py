#!/usr/bin/env python
#####################################################################
# This script presents different buffers and formats.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../examples/config/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.

# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from random import choice
from vizdoom import *

import cv2

game = DoomGame()

# Use other config file if you wish.
game.load_config("../../examples/config/deadly_corridor.cfg")
game.set_render_hud(False)
game.set_console_enabled(True)
# game.set_window_visible(False)

# Just umcomment desired format for screen buffer (and map buffer).
# The last uncommented will be applied.
# Formats with C (CRCGCB and CBCGCR) were ommited cause they are not cv2 friendly.
# Default format is ScreenFormat.CRCGCB.

game.set_screen_format(ScreenFormat.RGB24)
# game.set_screen_format(ScreenFormat.ARGB32)
# game.set_screen_format(ScreenFormat.GRAY8)

# game.set_screen_format(ScreenFormat.BGR24)
# game.set_screen_format(ScreenFormat.RGBA32)
# game.set_screen_format(ScreenFormat.BGRA32)
# game.set_screen_format(ScreenFormat.ABGR32)

# Raw Doom buffer with palette's values. This one makes no sense in particular
# game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

# Sets resolution for all buffers.
game.set_screen_resolution(ScreenResolution.RES_320X240)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of he current episode/level .
game.set_map_buffer_enabled(True)

game.set_mode(Mode.PLAYER)
game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

episodes = 10
# sleep time in ms
sleep_time = 28

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():
        # Gets the state and possibly to something with it
        s = game.get_state()

        # Display all the buffers here!

        # Just screen buffer, given in selected format. This buffer is always available.
        screen = s.screen_buffer
        cv2.imshow('ViZDoom Screen Buffer', screen)

        # Depth buffer, always in 8-bit gray channel format.
        # This is most fun. It looks best if you inverse colors.
        depth = s.depth_buffer
        if depth is not None:
            cv2.imshow('ViZDoom Depth Buffer', depth)

        # Labels buffer, always in 8-bit gray channel format.
        # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
        # Labels data are available in s.labels, also see labels.py example.
        labels = s.labels_buffer
        if labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', labels)

        # Map buffer, in the same format as screen buffer.
        # Shows top down map of the current episode/level.
        map = s.map_buffer
        if map is not None:
            cv2.imshow('ViZDoom Map Buffer', map)

        cv2.waitKey(sleep_time)

        v = s.game_variables
        r = game.make_action(choice(actions))

        print("State #" + str(s.number))
        print("=====================")

    print("Episode finished!")
    print("************************")

cv2.destroyAllWindows()