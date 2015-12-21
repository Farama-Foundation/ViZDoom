#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVar
from vizia import ScreenFormat
from random import choice


from time import sleep
from time import time
import cv2
import numpy as np
x = 120
y = 90
def setup_vizia():

	game = DoomGame()

	#available resolutions: 40x30, 60x45, 80x60, 100x75, 120x90, 160x120, 200x150, 320x240, 640x480
	game.set_screen_resolution(x,y)
	#This doesn't work
	game.set_screen_format(ScreenFormat.GRAY8)

	game.set_doom_game_path("../../bin/viziazdoom")
	game.set_doom_iwad_path("../../scenarios/doom2.wad")
	game.set_doom_file_path("../../scenarios/s1_b.wad")
	game.set_doom_map("map01")
	game.set_episode_timeout(200)

	game.set_living_reward(-1)
	game.set_render_hud(False)	
	game.set_render_crosshair(False)
	game.set_render_weapon(False)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.ATTACK)
	game.set_visible_window(False)
	game.set_action_interval(4)
	game.init()

	return game

#game.add_state_available_var(GameVar.AMMO1);

game = setup_vizia()


actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.0
start = time()
#for i in range(iters):
for i in range(iters):

	if game.is_episode_finished():
		#print game.get_summary_reward()
		print game.get_summary_reward()
		game.new_episode()
		sleep(1)
	s = game.get_state()
	#img = np.ma.average(s.image_buffer,axis=0, weights=[0.2989,0.5870,0.1140])	
	
	img = s.image_buffer
	img = np.float32(img)/255.0
	img =img.reshape(y,x)
	img[ img>0.2 ] = 1.0
	img = cv2.resize(img,(60,45))
	
	img = img.repeat(10,axis=0).repeat(10,axis=1)

	cv2.imshow('image',img)
	cv2.waitKey(1000) 
	r = game.make_action(choice(actions))
	print r
	#game.get_state()
	
	if sleep_time>0:
		sleep(sleep_time)
	#print "reward:",r
end=time()
t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)


game.close()


    
