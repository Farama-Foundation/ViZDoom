#!/usr/bin/python


from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("config_multi.properties")

game.add_custom_game_arg("-host")
game.add_custom_game_arg("2")
game.add_custom_game_arg("-deathmatch")
game.add_custom_game_arg("-warp")
game.add_custom_game_arg("-01")
game.set_mode(Mode.ASYNC_SPECTATOR)

game.init()

	
while not game.is_episode_finished():	
	s = game.get_state()
	img = s.image_buffer
	misc = s.game_variables

	game.advance_action()
	a = game.get_last_action()
	r = game.get_last_reward()
		
	print("state #"+str(s.number))
	print("game variables: ", misc)
	print("action:", a)
	print("reward:", r)
	print("=====================")
		
	
print("episode finished!")
print("summary reward:", game.get_summary_reward())
print("************************")

game.close()