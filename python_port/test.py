#!/usr/bin/python

import api
import numpy as np
import time 

def api_init_wrapper(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo):
    api.init(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo)

game_args = dict()
game_args['x'] = 320
game_args['y'] = 200
game_args['hit_reward'] = 1.01
game_args['max_moves'] = 50
# should be positive cause it's treated as a penalty
game_args['miss_penalty'] = 0.05
# should be negative cause it's treated as a reward
game_args['living_reward'] = -0.01
game_args['random_background'] = False
game_args['ammo'] = np.inf


game_args['x'] = game_args['x'] +(game_args['x']+1)%2
game_args['y'] = game_args['y'] +(game_args['y']+1)%2

print "Starting ..."
api_init_wrapper(**game_args)
print "init ok"
state_format = api.get_state_format()
correct_state_format = ((game_args['y'],game_args['x']),0) 

if state_format == correct_state_format:
	print "state_format ok"
else:
	print "state_format not ok"
	print "should be:", correct_state_format
	print "is:",state_format

steps = 45000

start = time.time()

action = [False,False,False]
for i in range(steps):
	if api.is_finished():
		api.new_episode();
	state = api.get_state()[0].copy()
	api.make_action(action)
	
end = time.time()

print "time: ", round(end - start,3), "s"
print "fps: ", round(steps/(end-start),2)