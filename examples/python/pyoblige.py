#!/usr/bin/env python3

#####################################################################
# This script presents how to use the environment with PyOblige.
# https://github.com/mwydmuch/PyOblige
#####################################################################

from __future__ import print_function
from time import sleep
import os

import vizdoom as vzd
from argparse import ArgumentParser
import oblige

DEFAULT_CONFIG = "../../scenarios/oblige.cfg"
DEFAULT_SEED = 666
DEFAULT_OUTPUT_FILE = "test.wad"

if __name__ == "__main__":
    parser = ArgumentParser("An example showing how to generate maps with PyOblige.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    parser.add_argument("-s", "--seed",
                        default=DEFAULT_SEED,
                        type=int,
                        help="Number of iterations(actions) to run")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Use verbose mode during map generation.")
    parser.add_argument("-o", "--output_file",
                        default=DEFAULT_OUTPUT_FILE,
                        help="Where the wad file will be created.")
    parser.add_argument("-x", "--exit",
                        action="store_true",
                        help="Do not test the wad, just leave after generation.")

    args = parser.parse_args()

    game = vzd.DoomGame()
    # Use your config
    game.load_config(args.config)
    game.set_doom_map("map01")
    game.set_doom_skill(3)

    # Create Doom Level Generator instance and set optional seed.
    generator = oblige.DoomLevelGenerator()
    generator.set_seed(args.seed)

    # Set generator configs, specified keys will be overwritten.
    generator.set_config({
        "size": "micro",
        "health": "more",
        "weapons": "sooner"})

    # There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
    generator.set_config(oblige.childs_play_wad)

    # Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
    generator.set_config({"length": "few"})

    # Generate method will return number of maps inside wad file.
    wad_path = args.output_file
    print("Generating {} ...".format(wad_path))
    num_maps = generator.generate(wad_path, verbose=args.verbose)
    print("Generated {} maps.".format(num_maps))

    if args.exit:
        exit(0)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(args.output_file)

    # Sets up game for spectator (you)
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    game.init()

    # Play as many episodes as maps in the new generated WAD file.
    episodes = num_maps

    # Play until the game (episode) is over.
    for i in range(1, episodes + 1):

        # Update map name
        print("Map {}/{}".format(i, episodes))
        map = "map{:02}".format(i)
        game.set_doom_map(map)
        game.new_episode()

        time = 0
        while not game.is_episode_finished():
            state = game.get_state()
            time = game.get_episode_time()

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("Kills:", game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        print("Items:", game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
        print("Secrets:", game.get_game_variable(vzd.GameVariable.SECRETCOUNT))
        print("Time:", time / 35, "s")
        print("************************")
        sleep(2.0)

    game.close()

    # Remove output of generator
    os.remove(wad_path)
    os.remove(wad_path.replace("wad", "old"))
    os.remove(wad_path.replace("wad", "txt"))
