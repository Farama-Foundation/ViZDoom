#!/usr/bin/python

from games import ShootingDotGame
import time
import api
import numpy as np

args = {}
args['x'] = 320
args['y'] = 200
args['hit_reward'] = 1.0
args['max_moves'] = 50
#should be positive cause it's treatet as a penalty
args['miss_penalty'] = 0.05
#should be negative cause it's treatet as a reward
args['living_reward'] = -0.01
args['random_background'] = False
args['ammo'] = np.inf
args['add_dimension'] = True
game = ShootingDotGame(**args)



steps = 45000
actions = [[True,False,False],[False,True,False],[False,False,True]]
action_ind = 0

start = time.time()
for i in range(steps):
	if game.is_finished():
		game.new_episode()
	s = game.get_state()[0].copy()
	game.make_action(actions[action_ind])
end = time.time()
print "Python dot_shooting:"
print "time:",round(end-start,4)
print "fps:",steps/(end-start)

api.init(args['x'],args['y'],args['random_background'],args['max_moves'],args['living_reward'],args['miss_penalty'],args['hit_reward'],args['ammo'])
start = time.time()
for i in range(steps):
	if api.is_finished():
		api.new_episode()
	s = api.get_state()[0].copy()
	api.make_action(actions[action_ind])

end = time.time()
print "C++ dot_shooting:"
print "time:",round(end-start,4)
print "fps:",steps/(end-start)