#!/usr/bin/python

from __future__ import print_function
from vizdoom import *
from random import choice

game = DoomGame()

# Use CIG example config or Your own.
game.load_config("../../examples/config/cig.cfg")

# Select game and map You want to use.
game.set_doom_game_path("../../scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences

game.set_doom_map("map01")  # Limited deathmatch.
#game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
#
game.add_game_args("-host 2 "               # This machine will function as a host for a multiplayer game with this many players (including this machine). It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                   "-deathmatch "           # Deathmatch rules are used for the game.
                   "+timelimit 10.0 "       # The game (episode) will end after this many minutes have elapsed.
                   "+sv_forcerespawn 1 "    # Players will respawn automatically after they die.
                   "+sv_noautoaim 1 "       # Autoaim is disabled for all players.
                   "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
                   "+sv_spawnfarthest 1 "   # Players will be spawned as far as possible from any other players.
                   "+vizdoom_nocheat 1")    # Disables depth buffer and the ability to use commands that could interfere with multiplayer game.

# Name Your AI.
game.add_game_args("+name AI")

# Multiplayer requires the use of asynchronous modes.
game.set_mode(Mode.ASYNC_PLAYER)

# game.set_window_visible(false)

game.init()

# Three example sample actions
actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

# Play until the game (episode) is over.
while not game.is_episode_finished():

    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

        # Or observe the game until automatic respawn.
        #game.advance_action();
        #continue;

    s = game.get_state()
    # Analyze the state.

    game.make_action(choice(actions))
    # Make your action.

    print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

game.close()
