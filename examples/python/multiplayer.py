#!/usr/bin/python

#use ./viziazdoom -host 2 -deathmatch -warp 01 to start game host
#or run host_multiplayer.py

from vizia import *
from random import choice
from time import sleep
from time import time



game = DoomGame()
game.load_config("config_multi.properties")

game.set_mode(Mode.ASYNC_PLAYER)
game.add_custom_game_arg("-join")
game.add_custom_game_arg("127.0.0.1")
game.init()


actions = [[1,0,0,0],[0,1,0,0],[0,0,0,1]]

iters = 10000
sleep_time = 0

for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		exit(0)

	s = game.get_state()

	print "gametic:", str(game.get_episode_time())
	print "state:", str(s.number)

	if sleep_time>0:
		sleep(sleep_time)

	r = game.make_action(choice(actions))

	print "reward:",r
	print "====================="

game.close()


