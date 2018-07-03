#!/usr/bin/env python3

#####################################################################
# This script presents how to use the environment with PyOblige.
# https://github.com/mwydmuch/PyOblige
#####################################################################

from __future__ import print_function
from time import sleep
import os

import vizdoom as vzd
import oblige


if __name__ == "__main__":
    game = vzd.DoomGame()

    # Use your config
    game.load_config("../../scenarios/oblige.cfg")
    game.set_doom_map("map01")
    game.set_doom_skill(3)

    # Create Doom Level Generator instance and set optional seed.
    generator = oblige.DoomLevelGenerator()
    generator.set_seed(666)

    # Set generator configs, specified keys will be overwritten.
    generator.set_config({"size": "micro", "health": "more", "weapons": "sooner"})

    # There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
    generator.set_config(oblige.test_wad)

    # Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
    generator.set_config({"length": "few"})

    # Generate method will return number of maps inside wad file.
    wad_path = "test.wad"
    maps = generator.generate(wad_path)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(wad_path)

    # Sets up game for spectator (you)
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    game.init()

    # Play as many episodes as maps in the new generated WAD file.
    episodes = maps

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