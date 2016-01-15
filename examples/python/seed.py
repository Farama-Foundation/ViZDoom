#!/usr/bin/python
from vizia import DoomGame
from vizia import ScreenResolution
from vizia import Mode

from random import choice

from time import sleep
from time import time

game = DoomGame()

game.load_config("config_basic.properties")              
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_mode(Mode.SPECTATOR)
game.init()

seed = 1234
game.set_seed(seed)

iters = 10000
for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()
		game.set_seed(seed)

	a = game.get_last_action()
	s = game.get_state()
	r = game.get_last_reward()
	
	print "state #" +str(s.number)
	print "action: ", a
	print "reward:",r
	print "====================="
	game.advance_action()

game.close()
TODO