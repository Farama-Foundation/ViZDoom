#!/usr/bin/python

from __future__ import print_function
from vizdoom import *
from random import choice

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
#from threading import Thread

def player1():
    game = DoomGame()

    #game.load_config('../config/basic.cfg')
    # or
    game.load_config('../config/multi_duel.cfg')
    game.add_game_args("-host 2 -deathmatch +timelimit 1.0 +sv_spawnfarthest 1")
    game.add_game_args("+name Player1")

    game.init()

    actions = [[True,False,False],[False,True,False],[False,False,True]]

    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()

        game.make_action(choice(actions))
        print("Player1 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

    game.close()

def player2():
    game = DoomGame()

    #game.load_config('../config/basic.cfg')
    # or
    game.load_config('../config/multi_duel.cfg')
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2")

    game.init()

    actions = [[True,False,False],[False,True,False],[False,False,True]]

    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()

        game.make_action(choice(actions))
        print("Player2 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

    game.close()

#p1 = Thread(target = player1)
#p1.start()

p1 = Process(target = player1)
p1.start()

player2()