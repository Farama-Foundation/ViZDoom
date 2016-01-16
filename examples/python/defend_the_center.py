#!/usr/bin/python
from vizia import DoomGame
from random import choice


from time import sleep
from time import time

game = DoomGame()
game.load_config("config_defend_the_center.properties")
game.init()
	
left = [True,False,False]
leftnshoot =[True, False, True]
actions = [left, leftnshoot]

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
	print "HP:", s.game_variables[0]
	print "AMMO:", s.game_variables[1]
	print "reward:",r
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    
