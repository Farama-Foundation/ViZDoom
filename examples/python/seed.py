#!/usr/bin/env python3

#####################################################################
# This script presents how to run deterministic episodes by setting
# seed. After setting the seed every episode will look the same (if
# agent will behave deterministicly of course).
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
#
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#
#####################################################################
from __future__ import print_function

import itertools as it
from random import choice
from time import sleep
from argparse import ArgumentParser
import vizdoom as vzd

DEFAULT_CONFIG = "../../scenarios/basic.cfg"

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to set seed to have deterministic episodes.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    parser.add_argument("-s", "--seed",
                        default=666,
                        type=int,
                        help="Seed for the random generator in DoomGame.")
    parser.add_argument("-e", "--per_episode",
                        action="store_true",
                        help="Set seed for every episode.")
    args = parser.parse_args()

    game = vzd.DoomGame()

    # Choose the scenario config file you wish to watch.
    # Don't load two configs cause the second will overrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.

    game.load_config(args.config)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Lets make episode shorter and observe starting position of Cacodemon.
    game.set_episode_timeout(50)

    if args.seed is not None:
        # Sets the seed. It can be change after init.
        game.set_seed(args.seed)

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

        # Seed can be changed anytime. It will take effect from next episodes.
        # game.set_seed(seed)
        if args.seed is not None and args.per_episode:
            game.set_seed(args.seed)
        game.new_episode()

        while not game.is_episode_finished():
            # Gets the state and possibly to something with it
            state = game.get_state()
            screen_buf = state.screen_buffer
            vars = state.game_variables

            # Check which action you chose!
            reward = game.make_action(choice(actions))

            print("State #" + str(state.number))
            print("Game Variables:", vars)
            print("Last Reward:", reward)
            print("Seed:", game.get_seed())
            print("=====================")

            # Sleep some time because processing is too fast to watch.
            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
