#!/usr/bin/env python3

#####################################################################
# This script presents how to join and play a singleplayer game,
# that can be hosted using ci_singleplayer_host.py script.
#####################################################################

from __future__ import print_function
from random import choice
from time import sleep
import os

from vizdoom import *
from oblige import *

game = DoomGame()

# Use your config
game.load_config("../../scenarios/cig.cfg")
game.set_doom_map("map01")
wad_path = "cig_singleplayer.wad"
game.set_doom_scenario_path(wad_path)

# Join existing game.
game.add_game_args("-join 127.0.0.1") # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)
#game.set_mode(Mode.PLAYER)
game.set_mode(Mode.ASYNC_PLAYER)

# Wait for wad generator
sleep(1)
while not os.path.exists(wad_path):
    sleep(1)

game.init()
i = 1
# Play until the game is over.
while True:
    print("Map {}".format(i))
    while not game.is_episode_finished():
        state = game.get_state()

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        if game.is_player_dead():
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

        print("State #" + str(state.number))
        print("Game variables: ", state.game_variables)
        print("Action:", last_action)
        print("Reward:", reward)
        print("=====================")

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

    game.new_episode()
    i += 1

game.close()
