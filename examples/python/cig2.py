#!/usr/bin/python

from __future__ import print_function
from vizia import *
from random import choice

game = DoomGame()
game.load_config("../../scenarios/config_cig2.properties")
game.set_doom_map("map02")

game.set_mode(Mode.ASYNC_PLAYER)
game.add_custom_game_arg("-join")
game.add_custom_game_arg("127.0.0.1")
game.set_window_visible(False)
game.init()
game.send_game_command("sv_forcerespawn 1");


actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

while not game.is_episode_finished():	
	game.make_action(choice(actions))
	print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
	while game.get_game_variable(GameVariable.DEAD):
		print("DEAD, waiting for respawn.")
		game.advance_action()


game.close()

