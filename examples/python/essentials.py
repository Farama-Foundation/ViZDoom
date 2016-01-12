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


game = DoomGame()

game.set_doom_iwad_path("../../scenarios/doom2.wad")
game.set_doom_file_path("../../scenarios/basic.wad")

game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

game.add_available_game_variable(GameVariable.AMMO1)

game.set_episode_timeout(200)
game.init()
	
actions = [[True,False,False],[False,True,False],[False,False,True]]


iters = 10000
sleep_time = 0.0

for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
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


    
