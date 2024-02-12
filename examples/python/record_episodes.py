#!/usr/bin/env python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

import os
from random import choice

import vizdoom as vzd


game = vzd.DoomGame()

# Use other config file if you wish.
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.set_episode_timeout(100)

# Record episodes while playing in 320x240 resolution without HUD
game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
game.set_render_hud(False)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.PLAYER)

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
    game.new_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished():
        s = game.get_state()

        a = choice(actions)
        r = game.make_action(choice(actions))

        print(f"State #{s.number}")
        print("Action:", a)
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print(f"Episode {i} finished. Saved to file episode{i}_rec.lmp")
    print("Total reward:", game.get_total_reward())
    print("************************\n")

game.new_episode()  # This is currently required to stop and save the previous recording.
game.close()

# New render settings for replay
game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
game.set_render_hud(True)

# Replay can be played in any mode.
game.set_mode(vzd.Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished():
        # Get a state
        s = game.get_state()

        # Use advance_action instead of make_action to proceed
        game.advance_action()

        # Retrieve the last actions and the reward
        a = game.get_last_action()
        r = game.get_last_reward()

        print(f"State #{s.number}")
        print("Action:", a)
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode", i, "finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

game.close()

# Delete recordings (*.lmp files).
for i in range(episodes):
    os.remove(f"episode{i}_rec.lmp")
