#!/usr/bin/python

#####################################################################
# This script presents how to run "basic" scenario.
# Configuration is loaded from "config_basic.properties" file.
# <iters> number of steps are performed that involve making an action,
# geting reward for the action and state (optionally resetting the 
# episode). Random action is chosen.
#
# To see the scenario description go to "../../scenarios/README"
# 
#####################################################################
from vizia import DoomGame
from random import choice

from time import sleep
from time import time

import cv2

game = DoomGame()
game.load_config("config_basic.properties")
game.init()

# MOVE_LEFT, MOVE_RIGHT and ATTACK
# There are 8 Combinations for 3 values but 3 most straightforward were chosen.
actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.1

for i in range(iters):

	if game.is_episode_finished():
		
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()

	# Get the state and possibly to something with it
	s = game.get_state()
	img = s.image_buffer
	misc = s.game_variables

	# Make a random action and save the reward.
	r = game.make_action(choice(actions))
	
	print "state #" +str(s.number)
	print "ammo:", misc[0]
	print "reward:",r
	print "====================="	
	# Sleep some time because processing is too fast to notice anything reasonable.
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()
