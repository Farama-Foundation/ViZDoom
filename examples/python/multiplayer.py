#use ./viziazdoom -host 2 -deathmatch -warp 01 to start game host

#!/usr/bin/python
from vizia import *

from random import choice

from time import sleep
from time import time

import cv2

def setup_vizia():

	game = DoomGame()

	game.set_screen_resolution(ScreenResolution.RES_640X480)
	game.set_screen_format(ScreenFormat.CRCGCB)

	game.set_doom_game_path("../viziazdoom")
	game.set_doom_iwad_path("../../scenarios/doom2.wad")

	game.set_living_reward(-1.0)
	game.set_death_penalty(100.0)

	game.set_render_hud(False)
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.ATTACK)
	game.add_available_button(Button.USE)

	game.set_window_visible(True)

	game.set_mode(Mode.ASYNC_PLAYER)

	game.add_custom_game_arg("-join")
	game.add_custom_game_arg("127.0.0.1")

	game.init()
	return game



game = setup_vizia()

actions = [[True,False,False,True],[False,True,False,True],[False,False,True,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.25

for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		#game.new_episode()

	s = game.get_state()

	print "gametic:", str(game.get_episode_time())
	print "state:", str(s.number)

	if sleep_time>0:
		sleep(sleep_time)

	r = game.make_action(shoot)

	print "reward:",r
	print "====================="

game.close()


