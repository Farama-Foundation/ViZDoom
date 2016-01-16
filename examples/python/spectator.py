#!/usr/bin/python
from vizia import DoomGame
from vizia import ScreenResolution
from vizia import Mode
from random import choice

from time import sleep
from time import time

game = DoomGame()

game.load_config("config_basic.properties")              
#game.load_config("config_deadly_corridor.properties")    
#game.load_config("config_defend_the_center.properties")  
#game.load_config("config_defend_the_line.properties")
#game.load_config("config_health_gathering.properties")
#game.load_config("config_my_way_home.properties")
#game.load_config("config_predict_position.properties")

game.set_screen_resolution(ScreenResolution.RES_800X450)
game.set_mode(Mode.SPECTATOR)
game.init()

iters = 10000
for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()

	a = game.get_last_action()
	s = game.get_state()
	r = game.get_last_reward()
	
	print "state #" +str(s.number)
	print "action: ", a
	print "reward:",r
	print "====================="
	game.advance_action()

game.close()
