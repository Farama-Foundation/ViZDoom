#!/usr/bin/python

#use ./viziazdoom -host 2 -deathmatch -warp 01 to start game host
#or run host_multiplayer.py to have a host

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time



game = DoomGame()
game.load_config("../../scenarios/config_cig2.properties")
game.set_doom_map("map03")

game.set_mode(Mode.ASYNC_PLAYER)
game.add_custom_game_arg("-join")
game.add_custom_game_arg("127.0.0.1")
game.set_window_visible(False)
game.init()


actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

while not game.is_episode_finished():	
	game.make_action(actions[0])
	print("frags:", game.get_game_variable(GameVariable.FRAGCOUNT))



game.close()


