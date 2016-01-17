#!/usr/bin/python

#####################################################################
# This script presents how to run some scenarios.
# Configuration is loaded from "config_<SCENARIO_NAME>.properties" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README"
# 
#####################################################################
from vizia import DoomGame, ScreenResolution
from random import choice
import itertools as it

from time import sleep
from time import time

import cv2

game = DoomGame()

# Choose the scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("config_basic.properties")
#game.load_config("config_deadly_corridor.properties")
#game.load_config("config_defend_the_center.properties")
#game.load_config("config_defend_the_line.properties")
#game.load_config("config_health_gathering.properties")
#game.load_config("config_my_way_home.properties")
#game.load_config("config_predict_position.properties")

# Makes the screen bigger to see more details.
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

# Creates all possible actions depending on how many buttons there are.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))


episodes = 10
sleep_time = 0.05

for i in range(episodes):

	# Not needed for the first episdoe but the loop is nicer.
	game.new_episode()
	while not game.is_episode_finished():
		print "Episode #" +str(i+1)

		# Gets the state and possibly to something with it
		s = game.get_state()
		img = s.image_buffer
		misc = s.game_variables

		# Makes a random action and save the reward.
		r = game.make_action(choice(actions))

		# Makes a "prolonged" action and skip frames:
		# skiprate = 3
		# r = game.make_action(choice(actions), skiprate)
		
		# The same could be achieved with:
		# game.set_action(choice(actions))
		# skiprate = 3
		# game.advance_action(True, True, skiprate)
		# r = game.get_last_reward()
		
		print "State #" +str(s.number)
		print "Game Variables:", misc
		print "Last Reward:",r
		print "====================="	

		# Sleep some time because processing is too fast to watch.
		if sleep_time>0:
			sleep(sleep_time)

	print "Episode finished!"
	print "Summary reward:", game.get_summary_reward()
	print "************************"


game.close()
