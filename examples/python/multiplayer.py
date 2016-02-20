#!/usr/bin/python

#use ./viziazdoom -host 2 -deathmatch -warp 01 to start game host
#or run host_multiplayer.py to have a host
#TODO add description

from __future__ import print_function
from vizia import *
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("../../scenarios/config_multi.properties")

game.set_mode(Mode.ASYNC_PLAYER)
game.add_custom_game_arg("-join")
game.add_custom_game_arg("127.0.0.1")
game.set_window_visible(False)
game.init()


actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]

episodes = 1

for i in range(episodes):
	
	while not game.is_episode_finished():	
		s = game.get_state()
		img = s.image_buffer
		misc = s.game_variables

		game.make_action(choice(actions))
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


