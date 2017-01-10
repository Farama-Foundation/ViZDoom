#!/usr/bin/env python

from __future__ import print_function

from random import choice
from vizdoom import *

game = DoomGame()

# Use CIG example config or your own.
game.load_config("../../scenarios/cig.cfg")

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")

game.set_mode(Mode.PLAYER)

# game.set_window_visible(False)

game.init()

# Three example sample actions
actions = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]

# Play with this many bots
bots = 7

# Run this many episodes
episodes = 10

for i in range(episodes):

    print("Episode #" + str(i + 1))

    # Add specific number of bots
    # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
    # edit this file to adjust bots).
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")

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

    print("Episode finished.")
    print("************************")

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()

game.close()
