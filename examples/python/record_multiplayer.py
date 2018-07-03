#!/usr/bin/python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# record multiplayer game and replay it with perfect accuracy.
#####################################################################

# WARNING:
# Due to the bug in build-in bots recording game with bots will result in the desynchronization of the recording.


from __future__ import print_function
from vizdoom import *
from random import choice
import os
from multiprocessing import Process


def player1():
    game = DoomGame()

    game.load_config('../../scenarios/multi_duel.cfg')
    game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1 ")
    game.add_game_args("+name Player1 +colorset 0")

    # Unfortunately multiplayer game cannot be recorded using new_episode() method, use this command instead.
    game.add_game_args("-record multi_rec.lmp")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()

        game.make_action(choice(actions))

    print("Game finished!")
    print("Player1 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
    game.close()


def player2():
    game = DoomGame()

    game.load_config('../../scenarios/multi_duel.cfg')
    game.set_window_visible(False)
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2 +colorset 3")

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()

        game.make_action(choice(actions))

    print("Player2 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
    game.close()


def replay_as_player2():
    game = DoomGame()
    game.load_config('../config/multi_duel.cfg')
    # At this moment ViZDoom will crash if there is no starting point - this is workaround for multiplayer map.
    game.add_game_args("-host 1 -deathmatch")

    game.init()

    # Replays episode recorded by player 1 from perspective of player2.
    game.replay_episode("multi_rec.lmp", 2)

    while not game.is_episode_finished():
        game.advance_action()

    print("Game finished!")
    print("Player1 frags:", game.get_game_variable(GameVariable.PLAYER1_FRAGCOUNT))
    print("Player2 frags:", game.get_game_variable(GameVariable.PLAYER2_FRAGCOUNT))
    game.close()

    # Delete multi_rec.lmp
    os.remove("multi_rec.lmp")


if __name__ == '__main__':
    print("\nRECORDING")
    print("************************\n")

    p1 = Process(target=player1)
    p1.start()

    player2()

    print("\nREPLAY")
    print("************************\n")

    replay_as_player2()
