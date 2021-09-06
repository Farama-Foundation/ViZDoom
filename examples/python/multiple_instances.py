#!/usr/bin/env python3

import os
from random import choice
from time import time
import vizdoom as vzd
from multiprocessing import Process
from threading import Thread

# Run this many episodes
episodes = 1
config = os.path.join(vzd.scenarios_path, "multi_duel.cfg")

def player1():
    game = vzd.DoomGame()

    game.load_config(config)
    game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1")
    game.add_game_args("+name Player1 +colorset 0")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(episodes):

        print("Episode #" + str(i + 1))

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
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2 +colorset 3")

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




if __name__ == '__main__':
    # Both Processes or Threads can be used to have many DoomGame instances running in parallel.
    # Because ViZDoom releases GIL, there is no difference in performance between Processes and Threads
    start = time()
    #p1 = Process(target=player1)
    p1 = Thread(target=player1)
    p1.start()
    player2()

    print("Finished", episodes, " after ", time() - start, "episodes.")
