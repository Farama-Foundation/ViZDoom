#!/usr/bin/python

import api
import numpy as np

def api_init_wrapper(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo):
    api.init(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo)

game_args = dict()
game_args['x'] = 7
game_args['y'] = 5
game_args['hit_reward'] = 1.01
game_args['max_moves'] = 50
# should be positive cause it's treated as a penalty
game_args['miss_penalty'] = 0.05
# should be negative cause it's treated as a reward
game_args['living_reward'] = -0.01
game_args['random_background'] = False
game_args['ammo'] = np.inf

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

api.new_episode()
print api.is_finished()
print api.get_state()
