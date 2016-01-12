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
	game.set_screen_format(ScreenFormat.CRCGCB)

	game.set_doom_game_path("./viziazdoom")
	game.set_doom_iwad_path("../../scenarios/doom2.wad")
	game.set_doom_file_path("../../scenarios/basic.wad")
	game.set_doom_map("map01")

	game.set_episode_timeout(300)
	game.set_living_reward(-1.0)

	game.set_render_hud(False)	
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.ATTACK)

	game.set_window_visible(True)
	game.add_available_game_variable(GameVariable.AMMO1)

	game.set_doom_skill(1)

	game.init()
	return game

	

game = setup_vizia()


actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.15

for i in range(iters):

	if game.is_episode_finished():
		
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.close()
		exit(0)
		game.new_episode()

	s = game.get_state()
	r = game.make_action(choice(actions))
	
	print "state #" +str(s.number)
	print "ammo:", s.game_variables[0]
	print "reward:",r
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    
