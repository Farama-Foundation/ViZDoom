#!/usr/bin/env python3

#####################################################################
# This script presents labels buffer that shows only visible game objects
# (enemies, pickups, exploding barrels etc.), each with unique label.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from random import choice
import vizdoom as vzd
import numpy as np
from argparse import ArgumentParser

import cv2

DEFAULT_CONFIG = "../../scenarios/deadly_corridor.cfg"
if __name__ =="__main__":
    parser = ArgumentParser("ViZDoom example showing how to use labels and labels buffer.")
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
    game.set_render_hud(False)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables labeling of the in game objects.
    game.set_labels_buffer_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10

    # Sleep time between actions in ms
    sleep_time = 28


    # Prepare some colors and drawing function
    # Colors in in BGR order
    doom_red_color = [0, 0, 203]
    doom_blue_color = [203, 0, 0]

    def draw_bounding_box(buffer, x, y, width, height, color):
        for i in range(width):
            buffer[y, x + i, :] = color
            buffer[y + height, x + i, :] = color

        for i in range(height):
            buffer[y + i, x, :] = color
            buffer[y + i, x + width, :] = color

    def color_labels(labels):
        """
        Walls are blue, floor/ceiling are red (OpenCV uses BGR).
        """
        tmp = np.stack([labels] * 3, -1)
        tmp[labels == 0] = [255, 0, 0]
        tmp[labels == 1] = [0, 0, 255]

        return tmp


    for i in range(episodes):
        print("Episode #" + str(i + 1))
        seen_in_this_episode = set()

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Labels buffer, always in 8-bit gray channel format.
            # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
            # Labels data are available in state.labels.
            labels = state.labels_buffer
            if labels is not None:
                cv2.imshow('ViZDoom Labels Buffer', color_labels(labels))

            # Screen buffer, given in selected format. This buffer is always available.
            # Using information from state.labels draw bounding boxes.
            screen = state.screen_buffer
            for l in state.labels:
                if l.object_name in ["Medkit", "GreenArmor"]:
                    draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_blue_color)
                else:
                    draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_red_color)
            cv2.imshow('ViZDoom Screen Buffer', screen)

            cv2.waitKey(sleep_time)

            game.make_action(choice(actions))

            print("State #" + str(state.number))
            print("Player position X:", state.game_variables[0], "Y:", state.game_variables[1], "Z:", state.game_variables[2])
            print("Labels:")

            # Print information about objects visible on the screen.
            # object_id identifies specific in game object.
            # object_name contains name of object.
            # value tells which value represents object in labels_buffer.
            for l in state.labels:
                seen_in_this_episode.add(l.object_name)
                # print("---------------------")
                print("Label:", l.value, "object id:", l.object_id, "object name:", l.object_name)
                print("Object position x:", l.object_position_x, "y:", l.object_position_y, "z:", l.object_position_z)

                # Other available fields:
                #print("Object rotation angle", l.object_angle, "pitch:", l.object_pitch, "roll:", l.object_roll)
                #print("Object velocity x:", l.object_velocity_x, "y:", l.object_velocity_y, "z:", l.object_velocity_z)
                print("Bounding box: x:", l.x, "y:", l.y, "width:", l.width, "height:", l.height)

            print("=====================")

        print("Episode finished!")

        print("=====================")

        print("Seen in this episode:")
        for l in seen_in_this_episode:
            print(l)

        print("************************")

    cv2.destroyAllWindows()
