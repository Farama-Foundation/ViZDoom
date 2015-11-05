#!/usr/bin/python

import numpy as np
from qengine import QEngine
from games import ShootingDotGame
from evaluators import MLPEvaluator
import random 
from time import time
from time import sleep
import itertools as it

def actions_generator(game):
	n = game.get_action_format()
	actions = []
	for perm in it.product([False,True],repeat= n):
		actions.append(perm)

	return actions



game_args = {}
game_args['width'] = 7
game_args['height'] = 1
game_args['hit_reward'] = 1.0
game_args['max_moves'] = 50
game_args['miss_penalty'] = 0 
game_args['living_reward'] = -0.05
game_args['random_background'] = False
game_args['ammo'] = 50


engine_args = {}
engine_args["history_length"] = 1
engine_args["bank_capacity"] = 1000
engine_args["evaluator"] = MLPEvaluator
engine_args["game"] = ShootingDotGame(**game_args)
engine_args['start_epsilon'] = 1.0
engine_args['epsilon_decay_start_step'] = 100000
engine_args['epsilon_decay_steps'] = 5000000
engine_args['actions_generator'] = actions_generator
engine_args['update_frequency'] = 250
engine_args['batch_size'] = 250
engine_args['gamma'] = 0.7

engine = QEngine(**engine_args)

epochs = 100 
training_episodes_per_epoch = 500
test_episodes_per_epoch = 50

overall_start = time()
print "Learning..."

for epoch in range(epochs):
	engine.learning_mode = True
	rewards = []
	start = time()
	for episode in range(training_episodes_per_epoch):
		r = engine.run_episode()
		rewards.append(r)
		
	end = time()
	print "\nEpoch",epoch
	print "Train:"
	print engine.get_actions_stats(True)
	print "steps:",engine._steps, ", mean:", np.mean(rewards), "eps:",engine._epsilon
	print "t:",round(end-start,2)

	#learning off
	if test_episodes_per_epoch >0:
		engine.learning_mode = False
		rewards = []
		start = time()
		for episode in range(test_episodes_per_epoch):
			r = engine.run_episode()
			rewards.append(r)
			
		end = time()
		print "Test"
		print engine.get_actions_stats(clear = True,norm = False)
		print "steps:",engine._steps, ", mean:", np.mean(rewards)
overall_end = time()

print "Elapsed time:",overall_end-overall_start