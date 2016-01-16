#!/usr/bin/python
from vizia import DoomGame
from random import choice
from vizia import ScreenResolution as res


from time import time

game = DoomGame()
game.load_config("config_basic.properties")
game.set_screen_resolution(res.RES_320X240)
game.set_window_visible(False)
game.init()

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


    
