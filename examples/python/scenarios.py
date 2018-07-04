#!/usr/bin/env python3

#####################################################################
# This script presents how to run some scenarios.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

import itertools as it
from random import choice
from time import sleep
import vizdoom as vzd
from argparse import ArgumentParser

DEFAULT_CONFIG = "../../scenarios/rocket_basic.cfg"
if __name__ == "__main__":

    parser = ArgumentParser("ViZDoom scenarios example.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()
    game = vzd.DoomGame()

    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overwrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.
    game.load_config(args.config)

    # Makes the screen bigger to see more details.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()

    # Creates all possible actions depending on how many buttons there are.
    actions_num = game.get_available_buttons_size()
    actions = []
    for perm in it.product([False, True], repeat=actions_num):
        actions.append(list(perm))

    episodes = 10
    sleep_time = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state and possibly to something with it
            state = game.get_state()

            # Makes a random action and save the reward.
            reward = game.make_action(choice(actions))

            print("State #" + str(state.number))
            print("Game Variables:", state.game_variables)
            print("Performed action:", game.get_last_action())
            print("Last Reward:", reward)
            print("=====================")

            # Sleep some time because processing is too fast to watch.
            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished!")
        print("total reward:", game.get_total_reward())
        print("************************")
