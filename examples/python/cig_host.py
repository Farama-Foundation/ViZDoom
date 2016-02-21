#!/usr/bin/python

from __future__ import print_function
from vizia import *

game = DoomGame()
game.load_config("../../scenarios/config_cig1.properties")
game.set_doom_map("map01")

game.add_custom_game_arg("-host")
game.add_custom_game_arg("2")
game.add_custom_game_arg("-deathmatch")
game.set_mode(Mode.ASYNC_SPECTATOR)
game.init()
game.send_game_command("sv_forcerespawn 1");

while True:#not game.is_episode_finished():	
	game.advance_action()
	print(game.get_episode_time()," Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

	while game.get_game_variable(GameVariable.DEAD):
		print("DEAD, waiting for respawn.")
		game.advance_action()

