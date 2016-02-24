#!/usr/bin/python

from __future__ import print_function
from vizia import *

game = DoomGame()
game.load_config("../config/cig1.cfg")
game.set_doom_map("map01")

game.add_game_args("-host")
game.add_game_args("2")
game.add_game_args("-deathmatch")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()
game.send_game_command("sv_forcerespawn 1");

while not game.is_episode_finished():	
	game.advance_action()
	print(game.get_episode_time()," Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

	while game.get_game_variable(GameVariable.DEAD):
		print("DEAD, waiting for respawn.")
		game.advance_action()

