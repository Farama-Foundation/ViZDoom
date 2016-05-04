#!/usr/bin/python

#####################################################################
# This script tests performance in frames per second.
# Change iters, resolution, window visibility, use get_ state or not.
# It should give you some idea how fast the framework can work on
# your hardware. The test involes copying the state to make it more 
# simillar to any reasonable usage. Comment the line with get_state 
# to exclude copying process.
#####################################################################
from __future__ import print_function
from vizdoom import *
from random import choice
from vizdoom import ScreenResolution as res
from time import time

# Some options:
resolution =res.RES_320X240
screen_format = ScreenFormat.DEPTH_BUFFER8
iterations = 10000

game = DoomGame()
game.load_config("../../examples/config/basic.cfg")

game.set_screen_resolution(resolution)
game.set_screen_format(screen_format)
game.set_window_visible(False)

game.init()

actions = [[True,False,False],[False,True,False],[False,False,True]]
left = actions[0]
right = actions[1]
shoot = actions[2]
idle = [False,False,False]

iterations = 10000
start = time()

print("Checking FPS rating. It may take some time. Be patient.")

for i in range(iterations):

    if game.is_episode_finished():
        game.new_episode()

    # Copying happens here
    s = game.get_state()
    game.make_action(choice(actions))

end=time()
t = end-start
print("Results:")
print("Iterations:", iterations)
print("Resolution:", resolution)
print("time:",round(t,3))
print("fps: ",round(iterations/t,2))


game.close()


    
