#!/usr/bin/env python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

from __future__ import print_function

import os
from random import choice
from vizdoom import *

game = DoomGame()

# Use other config file if you wish.
game.load_config("../../scenarios/basic.cfg")
game.set_episode_timeout(100)

# Record episodes while playing in 320x240 resolution without HUD
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.set_render_hud(False)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(Mode.PLAYER)

game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

# Run and record this many episodes
episodes = 2

# Recording
print("\nRECORDING EPISODES")
print("************************\n")

for i in range(episodes):

    # new_episode can record the episode using Doom's demo recording functionality to given file.
    # Recorded episodes can be reconstructed with perfect accuracy using different rendering settings.
    # This can not be used to record episodes in multiplayer mode.
    game.new_episode("episode" + str(i) + "_rec.lmp")

    while not game.is_episode_finished():
        s = game.get_state()

        r = game.make_action(choice(actions))

        print("State #" + str(s.number))
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************\n")

game.close()

# New render settings for replay
game.set_screen_resolution(ScreenResolution.RES_800X600)
game.set_render_hud(True)

# Replay can be played in any mode.
game.set_mode(Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode("episode" + str(i) + "_rec.lmp")

    while not game.is_episode_finished():
        s = game.get_state()

        # Use advance_action instead of make_action.
        game.advance_action()

        r = game.get_last_reward()
        # game.get_last_action is not supported and don't work for replay at the moment.

        print("State #" + str(s.number))
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")

game.close()

# Delete recordings (*.lmp files).
for i in range(episodes):
    os.remove("episode" + str(i) + "_rec.lmp")
