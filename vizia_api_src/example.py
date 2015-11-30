#!/usr/bin/python
from vizia import ViziaGame
from vizia import Button
from vizia import GameVar
from random import choice

from time import sleep
from time import time
import cv2
import numpy as np


game = ViziaGame()

#available resolutions: 40x30, 60x45, 80x60, 100x75, 120x90, 160x120, 200x150, 320x240, 640x480
game.set_screen_resolution(320,0)

game.set_doom_game_path("zdoom")
game.set_doom_iwad_path("doom2.wad")
game.set_doom_file_path("end_test.wad")
game.set_doom_map("map01")
game.set_episode_timeout_in_doom_tics(200)

game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False);

game.add_available_button(Button.MOVELEFT)#moveleft
game.add_available_button(Button.MOVERIGHT)#moveright
game.add_available_button(Button.ATTACK)#attack

game.add_state_available_var(GameVar.AMMO1);

#exit(0)
game.init()


actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 1000
sleep_time = 0.01
#game.new_episode()
start = time()
for i in range(iters):
	if game.is_episode_finished():
		game.new_episode()
		print "new episode"
		sleep(2)

	r = game.make_action(choice(actions))
	
	print "reward:",r
	s = game.get_state()
	#if r!=0:
		#print "reward:",r
	sleep(0.01)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)


game.close()


    
