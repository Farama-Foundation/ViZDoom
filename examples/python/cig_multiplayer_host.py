#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

from __future__ import print_function
from vizdoom import *
from random import choice

game = DoomGame()

# Use CIG example config or your own.
game.load_config("../../scenarios/cig.cfg")

game.set_doom_map("map01")  # Limited deathmatch.
#game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
game.add_game_args("-host 8 "  
                   # This machine will function as a host for a multiplayer game with this many players (including this machine). 
                   # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                   "-deathmatch "           # Deathmatch rules are used for the game.
                   "+timelimit 10.0 "       # The game (episode) will end after this many minutes have elapsed.
                   "+sv_forcerespawn 1 "    # Players will respawn automatically after they die.
                   "+sv_noautoaim 1 "       # Autoaim is disabled for all players.
                   "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
                   "+sv_spawnfarthest 1 "   # Players will be spawned as far as possible from any other players.
                   "+sv_nocrouch 1 "        # Disables crouching.
                   "+viz_respawn_delay 10 " # Sets delay between respanws (in seconds).
                   "+viz_nocheat 1")        # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

# This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
#game.add_game_args("+viz_spectator 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name Host +colorset 0")

# During the competition, async mode will be forced for all agents.
#game.set_mode(Mode.PLAYER)
game.set_mode(Mode.ASYNC_PLAYER)

#game.set_window_visible(False)

game.init()

# Three example sample actions
actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

# Play until the game (episode) is over.
while not game.is_episode_finished():

    # Get the state.
    s = game.get_state()

    # Analyze the state.

    # Make your action.
    game.make_action(choice(actions))

    # Check if player is dead
    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

game.close()
