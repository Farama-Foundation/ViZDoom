#!/usr/bin/python

#use multiplayer.py to have the second player

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("../../scenarios/config_cig1.properties")
game.set_doom_map("map01")

game.add_custom_game_arg("-host")
game.add_custom_game_arg("2")
game.add_custom_game_arg("-deathmatch")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()

	

while not game.is_episode_finished():	
	game.advance_action()
	print("frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
game.close()