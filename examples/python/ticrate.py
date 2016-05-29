#!/usr/bin/python

from __future__ import print_function
from vizdoom import *
from random import choice
from multiprocessing import Process

def play(game):
    game.init()

    actions = [[True,False,False],[False,True,False],[False,False,True]]
    episodes = 10

    for i in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            game.make_action(choice(actions))

    game.close()

def player1():
    game = DoomGame()

    game.load_config('../config/basic.cfg')
    game.set_mode(Mode.ASYNC_PLAYER)

    # Default Doom's ticrate is 35 per second, so this one will work 2 times faster.
    game.set_ticrate(70)

    play(game)

def player2():
    game = DoomGame()

    game.load_config('../config/basic.cfg')
    game.set_mode(Mode.ASYNC_PLAYER)

    # And this one will work 2 times slower.
    game.set_ticrate(17)

    play(game)


p1 = Process(target = player1)
p1.start()

player2()