#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVar
from vizia import ScreenFormat
from vizia import ScreenResolution
from vizia import GameMode

from random import choice

from time import sleep
from time import time

import cv2

def setup_vizia():

	game = DoomGame()

	game.set_game_mode(GameMode.SPECTATOR)
	game.set_screen_resolution(ScreenResolution.RES_960X720)
	game.set_screen_format(ScreenFormat.CRCGCB)

	game.set_doom_iwad_path("../../scenarios/doom2.wad")
	game.set_doom_file_path("../../scenarios/basic.wad")

	game.set_episode_timeout(100)

	game.set_living_reward(-1.0)
	game.set_death_penalty(300.0)

	game.set_render_hud(True)	
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.ATTACK)

	game.set_window_visible(True)

	game.set_doom_skill(1)

	game.init()
	return game

	

game = setup_vizia()


iters = 10000

for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()

	#not supported in python yet
	#a = game.get_last_action()
	s = game.get_state()
	r = game.get_last_reward()
	print "state #" +str(s.number)
	print "reward:",r
	print "====================="	
	game.advance_action()


game.close()


    
