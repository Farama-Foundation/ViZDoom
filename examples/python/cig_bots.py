#!/usr/bin/python

from __future__ import print_function
from vizdoom import *
from random import choice

game = DoomGame()

game.set_vizdoom_path("../../bin/vizdoom")

# Use CIG example config or Your own.
game.load_config("../../examples/config/cig.cfg")

# Select game and map You want to use.
game.set_doom_game_path("../../scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences

game.set_doom_map("map01")  # Limited deathmatch.
#game.set_doom_map("map02")  # Full deathmatch.

# Start multiplayer game only with Your AI (with options that will be used in the competition, details in cig_host example).
game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

# Name Your AI.
game.add_game_args("+name AI")

# Multiplayer requires the use of asynchronous modes, but when playing only with bots, synchronous modes can also be used.
game.set_mode(Mode.PLAYER)

# game.set_window_visible(false)

game.init()

# Three example sample actions
actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

# Add bots (file examples/bots.cfg must be placed in the same directory as the Doom executable file).
bots_number = 7
for i in range(bots_number):
    game.send_game_command("addbot")

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
