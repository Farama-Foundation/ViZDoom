#!/usr/bin/env python3

import os
from random import choice
from threading import Thread
from time import time

import vizdoom as vzd


# Run this many episodes
episodes = 1
config = os.path.join(vzd.scenarios_path, "multi_duel.cfg")
win_x = 100
win_y = 100


def player1():
    game = vzd.DoomGame()

    game.load_config(config)

    # Setup 2 players deathmatch game that will time out after 1 minute
    game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1")

    # Use additional arguments to set player name, color and window position
    game.add_game_args(f"+name Player1 +colorset 0 +win_x {win_x} +win_y {win_y}")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(episodes):
        print(f"Episode #{i + 1}")

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            game.make_action(choice(actions))

        print("Episode finished!")
        print("Player1 frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()


def player2():
    game = vzd.DoomGame()

    game.load_config(config)

    # Join existing game
    game.add_game_args("-join 127.0.0.1")

    # Use additional arguments to set player name, color and window position
    game.add_game_args(
        f"+name Player2 +colorset 3 +win_x {win_x + game.get_screen_width()} +win_y {win_y}"
    )

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(episodes):

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            game.make_action(choice(actions))

        print("Player2 frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()


if __name__ == "__main__":
    # Both Processes or Threads can be used to have many DoomGame instances running in parallel.
    # Because ViZDoom releases GIL, there is no/minimal difference in performance between Processes and Threads.
    start = time()
    # p1 = Process(target=player1)
    p1 = Thread(target=player1)
    p1.start()
    player2()

    print("Finished", episodes, "episodes after", time() - start, "seconds")
