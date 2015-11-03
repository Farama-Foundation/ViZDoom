#!/usr/bin/python

import numpy as np
from qengine import QEngine
from games import ShootingDotGame
from evaluators import CNNEvaluator

from time import time
from time import sleep

def generator(game):
	return [[0],[1],[2],[3]]

class AlwaysZeroEvaluator:
	def __init__(self,state_format, actions_number):
		None
	def learn(self, transitions):
		None
	def evaluate(self, state):
		return 0

game_args = {}
game_args['width'] = 5
game_args['height'] = 1
game_args['hit_reward'] = 1
game_args['max_moves'] = 600
game_args['miss_penalty'] = 0 
game_args['living_reward'] = -0.01
game_args['random_background'] = False

engine_args = {}
engine_args["history_length"] = 2
engine_args["bank_capacity"] = 10000
engine_args["evaluator"] = AlwaysZeroEvaluator
engine_args["game"] = ShootingDotGame(**game_args)
engine_args['epsilon_decay_start_step'] = 500000
engine_args['epsilon_decay_steps'] = 5000000
engine_args['actions_generator'] = generator

engine = QEngine(**engine_args)

engine.make_step()
