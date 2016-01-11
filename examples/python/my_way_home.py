#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVariable
from vizia import ScreenFormat
from vizia import ScreenResolution

from random import choice


from time import sleep
from time import time

import cv2

def setup_vizia():

	game = DoomGame()

	game.set_screen_resolution(ScreenResolution.RES_320X240)

	game.set_doom_iwad_path("../../scenarios/doom2.wad")
	game.set_doom_file_path("../../scenarios/my_way_home.wad")
	game.set_episode_timeout(2100)

	game.set_living_reward(-0.0001)

	game.set_render_hud(False)	
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.TURN_LEFT)
	game.add_available_button(Button.TURN_RIGHT)
	game.add_available_button(Button.MOVE_FORWARD)


	game.set_window_visible(True)

	game.init()
	
	return game

	

game = setup_vizia()

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


    
