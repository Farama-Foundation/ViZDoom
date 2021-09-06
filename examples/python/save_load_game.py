#!/usr/bin/env python3

#####################################################################
# This script demonstrates game save and load functionality.
#####################################################################

# WARNING:
# Please note that this feature is experimental and not well tested!

import itertools as it
import os
from random import choice
from time import sleep
import vizdoom as vzd

if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    game.load_config("../../scenarios/health_gathering_supreme.cfg")
    game.set_render_hud(True)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)

    # Creates all possible actions depending on how many buttons there are.
    actions_num = game.get_available_buttons_size()
    actions = []
    for perm in it.product([False, True], repeat=actions_num):
        actions.append(list(perm))

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    # Save after that many states
    save_after_steps = 50

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    print("Starting first episode...")

    # Starts a new episode.
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        reward = game.make_action(choice(actions))

        if state.number == save_after_steps:
            # The current game state will be save to the file,
            game.save("save.png") # Save files can be saved as png files

            print("\nGame saved!")
            print("Game variables:", state.game_variables)

        if sleep_time > 0:
            sleep(sleep_time)


    print("Starting second episode from saved game...")
    # It's not necessary, but we recommend calling new_episode before load.
    # Loading the game state does not reset the current episode state,
    # tic counter/time and total reward state keep their values.
    game.new_episode()
    game.load("save.png")
    # A new state is available after loading.
    state = game.get_state()

    # There can be small difference in some of the game variables
    print("\nGame loaded!")
    print("Game variables:", state.game_variables)

    while not game.is_episode_finished():
        state = game.get_state()
        reward = game.make_action(choice(actions))

        if sleep_time > 0:
            sleep(sleep_time)

    game.close()

    # Delete save file
    os.remove("save.png")