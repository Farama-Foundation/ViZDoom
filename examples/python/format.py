#!/usr/bin/python
from vizia import DoomGame
from random import choice
from vizia import ScreenFormat
from time import sleep
from time import time

import cv2

game = DoomGame()
game.load_config("config_basic.properties")
#game.set_window_visible(False)

# Just umcomment desired format. The last uncommented will be applied.
# Formats with C were ommited cause they are not cv2 friendly
game.set_screen_format(ScreenFormat.RGB24)
#game.set_screen_format(ScreenFormat.ARGB32)
#game.set_screen_format(ScreenFormat.GRAY8)

#Not working yet but will be soon
#game.set_screen_format(ScreenFormat.ZBUFFER8)

#These formats can be use bet they do not make much sense for cv2, you'll just get mixed up colors.
#game.set_screen_format(ScreenFormat.BGR24)
#game.set_screen_format(ScreenFormat.RGBA32) 
#game.set_screen_format(ScreenFormat.BGRA32) 
#game.set_screen_format(ScreenFormat.ABGR32)

#This one makes no sense especially :D
#game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000

for i in range(iters):

	if game.is_episode_finished():
		
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		game.new_episode()

	s = game.get_state()
	img = s.image_buffer
	
	# Gray8 shape is not cv2 compliant
	if game.get_screen_format() == ScreenFormat.GRAY8:
		img = img.reshape(img.shape[1],img.shape[2],1)

	# Display the image here!
	cv2.imshow('Doom Buffer',img)
	cv2.waitKey(13)

	r = game.make_action(choice(actions))
	

	print "state #" +str(s.number)
	print "ammo:", s.game_variables[0]
	print "reward:",r
	print "====================="	
	
	
cv2.destroyAllWindows()
game.close()
