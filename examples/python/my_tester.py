#!/usr/bin/python
from __future__ import print_function
from vizia import *
from random import choice
import itertools as it
from time import sleep

import cv2

game = DoomGame()

#game.load_config("config_basic.properties")
#game.load_config("config_deadly_corridor.properties")
#game.load_config("config_defend_the_center.properties")
#game.load_config("config_defend_the_line.properties")
game.load_config("config_health_gathering.properties")
#game.load_config("config_my_way_home.properties")
#game.load_config("config_predict_position.properties")

game.set_doom_file_path("../../scenarios/health_guided.wad")
game.set_screen_resolution(ScreenResolution.RES_640X480)

game.init()

actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))


episodes = 10
sleep_time = 0.028
skiprate = 8

for i in range(episodes):

	game.new_episode()
	while not game.is_episode_finished():
		s = game.get_state()
		img = s.image_buffer
		misc = s.game_variables

		r = game.make_action(actions[0], skiprate)


		
		print "State #" +str(s.number)
		print "Game Variables:", misc
		print "Performed action:",game.get_last_action()
		print "Last Reward:",r
		print "====================="	

		if sleep_time>0:
			sleep(sleep_time)

	print "Episode finished!"
	print "Summary reward:", game.get_summary_reward()
	print "************************"
	print "\n\n\n\n\n\n"
	sleep(2)
