#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVariable
from random import choice
from vizia import ScreenResolution


from time import time


def setup_vizia():

	game = DoomGame()

	game.set_screen_resolution(ScreenResolution.RES_320X240)
	game.set_window_visible(False)

	
	game.set_doom_iwad_path("../../scenarios/doom2.wad")
	game.set_doom_file_path("../../scenarios/s1_b.wad")
	game.set_episode_timeout(200)

	game.set_living_reward(-1)
	
	game.set_render_hud(False)	
	game.set_render_crosshair(False)
	game.set_render_weapon(True)
	game.set_render_decals(False)
	game.set_render_particles(False);

	game.add_available_button(Button.MOVE_LEFT)
	game.add_available_button(Button.MOVE_RIGHT)
	game.add_available_button(Button.ATTACK)
	
	game.init()

	return game

game = setup_vizia()



actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iters = 10000
sleep_time = 0.0
start = time()

print "\nChecking FPS rating. It may take some time. Be patient."

for i in range(iters):

	if game.is_episode_finished():		
		game.new_episode()
	
	s = game.get_state()
	r = game.make_action(choice(actions))
	
end=time()
t = end-start
print "Results:"
print "time:",round(t,3)
print "fps: ",round(iters/t,2)


game.close()


    
