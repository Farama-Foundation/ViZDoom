#!/usr/bin/python

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("../config/cig1.cfg")
#game.load_config("../config/cig2.cfg")

game.add_game_args("-host")
game.add_game_args("1")
game.add_game_args("-deathmatch")
game.add_game_args("-respawn")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()
game.send_game_command("sv_forcerespawn 1");
for i in range(7):
	game.send_game_command("addbot")

	

while not game.is_episode_finished():	
	game.advance_action()
	print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
	while game.get_game_variable(GameVariable.DEAD):
		print("DEAD, waiting for respawn.")
		game.advance_action()