#!/usr/bin/python

#use multiplayer.py to have the second player

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("config_cig1.properties")

game.add_custom_game_arg("-host")
game.add_custom_game_arg("4")
game.add_custom_game_arg("-deathmatch")
game.add_custom_game_arg("-warp")
game.add_custom_game_arg("-01")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()

	

while not game.is_episode_finished():	
	game.advance_action()
	print("frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
game.close()