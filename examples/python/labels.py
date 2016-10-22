#!/usr/bin/env python

#####################################################################
# This script presents labels buffer that shows only visible game objects
# (enemies, pickups, exploding barrels etc.), each with unique label.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../examples/config/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
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

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables labeling of the in game objects.
game.set_labels_buffer_enabled(True)

game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

episodes = 10

# Sleep time between actions in ms
sleep_time = 28

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():
        # Gets the state and possibly to something with it
        state = game.get_state()

        labels = state.labels_buffer
        if labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', labels)

        cv2.waitKey(sleep_time)

        game.make_action(choice(actions))

        print("State #" + str(state.number))
        print("Labels:")

        # Print information about objects visible on the screen.
        # object_id identifies specific in game object.
        # object_name contains name of object.
        # value tells which value represents object in labels_buffer.
        for l in state.labels:
            print("Object id: " + str(l.object_id) + " object name: " + l.object_name + " label: " + str(l.value))

        print("=====================")

    print("Episode finished!")
    print("************************")

cv2.destroyAllWindows()