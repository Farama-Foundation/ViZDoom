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
	game.set_doom_file_path("../../scenarios/health_gathering.wad")
	game.set_episode_timeout(2100)

	game.set_living_reward(0.25)
	game.set_death_penalty(100)

	game.set_render_hud(False)	
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.TURN_LEFT)
	game.add_available_button(Button.TURN_RIGHT)
	game.add_available_button(Button.MOVE_FORWARD)


	game.set_window_visible(True)
	game.add_available_game_variable(GameVariable.HEALTH)

	game.init()
	
	return game

	

game = setup_vizia()

left = [True,False,True]
right = [False,True,True]
forward =[False, False, True]
actions = [left, right, forward]

iters = 10000
sleep_time = 0.5


for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		sleep(1)
		game.new_episode()

	skiprate = 4
	s = game.get_state()

	game.set_action(choice(actions))
	game.advance_action(True,True,skiprate)
	r = game.get_last_reward()

	print "state #" +str(s.number)
	print "HP:", s.game_variables[0]
	print "reward:",r
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    
