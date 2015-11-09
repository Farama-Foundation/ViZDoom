#!/usr/bin/python

from games import ShootingDotGame
import time
import numpy as np

game_args = {}
game_args['width'] = 320
game_args['height'] = 200
game_args['hit_reward'] = 1.0
game_args['max_moves'] = 50
#should be positive cause it's treatet as a penalty
game_args['miss_penalty'] = 0.05
#should be negative cause it's treatet as a reward
game_args['living_reward'] = -0.01
game_args['random_background'] = True
game_args['ammo'] = np.inf
game_args['add_dimension'] = True
game = ShootingDotGame(**game_args)


steps = 45000
start = time.time()
action = [0,1,0]
for i in range(steps):
	if game.is_finished():
		game.new_episode()
	s = game.get_state()
	game.make_action(action)

end = time.time()
print "time:",round(end-start,4)
print "fps:",steps/(end-start)

