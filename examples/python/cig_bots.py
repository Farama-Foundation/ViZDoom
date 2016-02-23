#!/usr/bin/python

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("../../scenarios/config_cig1.properties")
game.set_doom_map("map01_bt")

game.add_custom_game_arg("-host")
game.add_custom_game_arg("1")
game.add_custom_game_arg("-deathmatch")
game.add_custom_game_arg("-respawn")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()
game.send_game_command("sv_forcerespawn 1");

	

while not game.is_episode_finished():	
	game.advance_action()
	print("Overall kills:", game.get_game_variable(GameVariable.KILLCOUNT))
	print("Your frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
	print()

	while game.get_game_variable(GameVariable.DEAD):
		print("DEAD, waiting for respawn.")
		game.advance_action()