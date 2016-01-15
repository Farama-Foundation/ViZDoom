#!/usr/bin/python
from vizia import DoomGame
from random import choice
from time import sleep
from time import time

game = DoomGame()
game.load_config("config_my_way_home.properties")
game.init()

left = [True,False,False]
right = [False,True,False]
forward =[False, False, True]
actions = [left, right, forward]

iters = 10000
sleep_time = 0.05


for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		sleep(1)
		game.new_episode()

	s = game.get_state()
	r = game.make_action(choice(actions))

	print "state #" +str(s.number)
	print "reward:",r
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    
