#!/usr/bin/env python3

#####################################################################
# This script presents how to make use of game variables to implement
# shaping using health_guided.wad scenario
# Health_guided scenario is just like health_gathering
# (see "../../scenarios/README.md") but for each collected medkit global
# variable number 1 in acs script (coresponding to USER1) is increased
# by 100.0. It is not considered a part of reward but will possibly
# reduce learning time.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
#####################################################################
from __future__ import print_function

import itertools as it
from random import choice
from time import sleep
from argparse import ArgumentParser
import vizdoom as vzd

DEFAULT_CONFIG = "../../scenarios/health_gathering.cfg"

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use shaping for health gathering scenario.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.

    game.load_config(args.config)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.init()

    # Creates all possible actions.
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

        # Use this to remember last shaping reward value.
        last_total_shaping_reward = 0

        while not game.is_episode_finished():

            # Gets the state and possibly to something with it
            state = game.get_state()

            # Makes a random action and save the reward.
            reward = game.make_action(choice(actions))

            # Retrieve the shaping reward
            fixed_shaping_reward = game.get_game_variable(vzd.GameVariable.USER1)  # Get value of scripted variable
            shaping_reward = vzd.doom_fixed_to_double(
                fixed_shaping_reward)  # If value is in DoomFixed format project it to double
            shaping_reward = shaping_reward - last_total_shaping_reward
            last_total_shaping_reward += shaping_reward

            print("State #" + str(state.number))
            print("Health: ", state.game_variables[0])
            print("Last Reward:", reward)
            print("Last Shaping Reward:", shaping_reward)
            print("=====================")

            # Sleep some time because processing is too fast to watch.
            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
