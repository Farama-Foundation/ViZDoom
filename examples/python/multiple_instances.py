#!/usr/bin/env python3

from __future__ import print_function

from random import choice
from vizdoom import *
# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread

# Run this many episodes
episodes = 10


def player1():
    game = DoomGame()

    # game.load_config('../../scenarios/basic.cfg')
    # or
    game.load_config('../../scenarios/multi_duel.cfg')
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
        print("Player1 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()


def player2():
    game = DoomGame()

    # game.load_config('../config/basic.cfg')
    # or
    game.load_config('../../scenarios/multi_duel.cfg')
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2 +colorset 3")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(episodes):

        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()

            game.make_action(choice(actions))

        print("Player2 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()


# p1 = Thread(target = player1)
# p1.start()

if __name__ == '__main__':
    p1 = Process(target=player1)
    p1.start()
    player2()

    print("Done")
