#!/usr/bin/python

import api
import numpy as np
from games import ShootingDotGame

def api_init_wrapper(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo):
    api.init(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo)

game_args = dict()
game_args['x'] = 5
game_args['y'] = 1
game_args['hit_reward'] = 1.01
game_args['max_moves'] = 50
# should be positive cause it's treated as a penalty
game_args['miss_penalty'] = 0.05
# should be negative cause it's treated as a reward
game_args['living_reward'] = -0.01
game_args['random_background'] = False
game_args['ammo'] = np.inf

game_s = ShootingDotGame(**game_args)
api_init_wrapper(**game_args)
game_api = api

iters = 10000
x_stats = np.zeros([game_args['x']], dtype = np.float32)
for i in range(iters):
	game_api.new_episode()
	st = game_api.get_state()
	x_stats += st[0][0]

print x_stats/x_stats.sum()


x_stats = np.zeros([game_args['x']], dtype = np.float32)
for i in range(iters):
	game_s.new_episode()
	st = game_s.get_state()
	x_stats += st[0][0]

print x_stats/x_stats.sum()


