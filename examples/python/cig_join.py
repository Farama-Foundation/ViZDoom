#!/usr/bin/python

from __future__ import print_function
from vizia import *
from random import choice

game = DoomGame()
game.load_config("../config/cig.cfg")
game.set_doom_map("map01")
#game.set_doom_map("map02")

game.set_mode(Mode.ASYNC_PLAYER)
game.add_game_args("-join")
game.add_game_args("127.0.0.1")
game.set_window_visible(False)
game.init()

# Three sample actions
actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

while not game.is_episode_finished():	
	if game.is_player_dead():
		game.advance_action(
		continue
	game.make_action(choice(actions))
	print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
		

