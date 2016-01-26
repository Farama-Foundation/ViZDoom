#!/usr/bin/python

#####################################################################
# This script presents how to configure	the engine without loading
# any configuration files.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README"
# 
#####################################################################

from vizia import DoomGame
from vizia import Button
from vizia import GameVariable
from vizia import ScreenFormat
from vizia import ScreenResolution

from random import choice

from time import sleep
from time import time

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


actions = [[True,False,False],[False,True,False],[False,False,True]]


episodes = 10
sleep_time = 0.028

for i in range(episodes):
	print "Episode #" +str(i+1)
	
	# Not needed for the first episdoe but the loop is nicer.
	game.new_episode()
	while not game.is_episode_finished():

		# Gets the state and possibly to something with it
		s = game.get_state()
		img = s.image_buffer
		misc = s.game_variables

		# Makes a random action and save the reward.
		r = game.make_action(choice(actions))

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