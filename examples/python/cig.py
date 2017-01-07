#!/usr/bin/env python

from __future__ import print_function
from vizdoom import *
from random import choice

game = DoomGame()

# Use CIG example config or your own.
game.load_config("../../scenarios/cig.cfg")

game.set_doom_map("map01")  # Limited deathmatch.
#game.set_doom_map("map02")  # Full deathmatch.

# Join existing game.
game.add_game_args("-join 127.0.0.1") # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

game.set_mode(Mode.ASYNC_PLAYER)

#game.set_window_visible(false)

game.init()

# Three example sample actions
actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

# Play until the game (episode) is over.
while not game.is_episode_finished():

    s = game.get_state()
    # Analyze the state.

    game.make_action(choice(actions))
    # Make your action.

    # Check if player is dead
    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

    print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

game.close()
