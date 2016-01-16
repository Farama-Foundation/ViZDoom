#!/usr/bin/python
from vizia import DoomGame
from vizia import GameVariable
from vizia import doom_fixed_to_float
from random import choice

from time import sleep
from time import time

game = DoomGame()
game.load_config("config_health_gathering.properties")
game.init()


left = [True, False, False]
right = [False, True, False]
forward =[False, False, True]
actions = [left, right, forward]

iters = 10000
sleep_time = 0.1


for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		sleep(1)
		game.new_episode()

	s = game.get_state()
	r = game.make_action(choice(actions))
	sr = doom_fixed_to_float(game.get_game_variable(GameVariable.USER1))

	print "state #" +str(s.number)
	print "HP:", s.game_variables[0]
	print "reward:",r
	print "summmary shaping reward:", sr
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    
