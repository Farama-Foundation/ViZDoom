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
game.set_screen_resolution(40,30)

game.set_doom_game_path("zdoom")
game.set_doom_iwad_path("doom2.wad")
game.set_doom_file_path("s1_b.wad")
game.set_doom_map("map01")
game.set_episode_timeout_in_doom_tics(200)

game.set_living_reward(-1)
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False);

game.add_available_button(Button.MOVELEFT)
game.add_available_button(Button.MOVERIGHT)
game.add_available_button(Button.ATTACK)

#game.add_state_available_var(GameVar.AMMO1);

game.init()

actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 1200
sleep_time = 0.01
start = time()
#for i in range(iters):
i =0
while True:
	if game.is_episode_finished():
		#print "summary reward:",game.get_summary_reward()
		print i
		i+=1
		game.new_episode()
	r = game.make_action(choice(actions))
	s = game.get_state()
	#print "reward:",r
	#sleep(0.01)
end=time()

t = end-start
print "time:",round(t,3)
print "fps: ",round(iters/t,2)


game.close()


    
