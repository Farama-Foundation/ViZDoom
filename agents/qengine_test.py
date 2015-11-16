#!/usr/bin/python

import numpy as np
from qengine import QEngine
from games import ShootingDotGame
from evaluators import MLPEvaluator
from evaluators import CNNEvaluator
from time import time
import itertools as it
import lasagne
import api

def api_init_wrapper(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo, add_dimension):
    api.init(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo)

def actions_generator(the_game):
    n = the_game.get_action_format()
    actions = []
    for perm in it.product([False, True], repeat=n):
        actions.append(perm)
    return actions


def create_mlp_evaluator(state_format, actions_number, batch_size, gamma):
    mlp_args = dict()
    mlp_args["gamma"] = gamma
    mlp_args["state_format"] = state_format
    mlp_args["actions_number"] = actions_number
    mlp_args["batch_size"] = batch_size
    network_args = dict(hidden_units=[500], learning_rate=0.01, hidden_layers=1)
    network_args["updates"] = lasagne.updates.nesterov_momentum
    mlp_args["network_args"] = network_args
    return MLPEvaluator(**mlp_args)


def create_cnn_evaluator(state_format, actions_number, batch_size, gamma):
    cnn_args = dict()
    cnn_args["gamma"] = gamma
    cnn_args["state_format"] = state_format
    cnn_args["actions_number"] = actions_number
    cnn_args["batch_size"] = batch_size
    network_args = dict(hidden_units=[500], learning_rate=0.01, hidden_layers=1)
    cnn_args["network_args"] = network_args

    network_args["updates"] = lasagne.updates.nesterov_momentum
    network_args["pool_size"] = [(2, 2), (2, 2)]
    network_args["num_filters"] = [16, 16]
    network_args["filter_size"] = [4, 4]
    return CNNEvaluator(**cnn_args)


game_args = dict()
game_args['x'] = 11
game_args['y'] = 5
game_args['hit_reward'] = 1.00
game_args['max_moves'] = 50
# should be positive cause it's treated as a penalty
game_args['miss_penalty'] = 0.05
# should be negative cause it's treated as a reward
game_args['living_reward'] = 0.0
game_args['random_background'] = False
game_args['ammo'] = np.inf
game_args['add_dimension'] = False

shooting_game = False
if shooting_game:
    game = ShootingDotGame(**game_args)
else:
    api_init_wrapper(**game_args)
    game = api

engine_args = dict()
engine_args["history_length"] = 1
engine_args["bank_capacity"] = 10000
engine_args["evaluator"] = create_mlp_evaluator
engine_args["game"] = game
engine_args['start_epsilon'] = 1.0
engine_args['epsilon_decay_start_step'] = 500000
engine_args['epsilon_decay_steps'] = 1000000
engine_args['actions_generator'] = actions_generator
engine_args['update_frequency'] = 4
engine_args['batch_size'] = 25
engine_args['gamma'] = 0.8

engine = QEngine(**engine_args)
# engine.online_mode = True

epochs = np.inf
training_episodes_per_epoch = 1000
test_episodes_per_epoch = 100
stop_mean = 1.0  # game.average_best_result()
overall_start = time()
print "Average best result:", round(game.average_best_result(), 4)
print "Learning..."

epoch = 0
while epoch < epochs:
    engine.learning_mode = True
    rewards = []
    start = time()
    for episode in range(training_episodes_per_epoch):
        r = engine.run_episode()
        rewards.append(r)

    end = time()
    print "\nEpoch", epoch
    print "Train:"
    print engine.get_actions_stats(True)
    print "steps:", engine.get_steps(), ", mean:", np.mean(rewards), ", max:", np.max(
        rewards), "eps:", engine.get_epsilon()
    print "t:", round(end - start, 2)
    # learning off
    if test_episodes_per_epoch > 0:
        engine.learning_mode = False
        rewards = []
        start = time()
        for test_episode in range(test_episodes_per_epoch):
            r = engine.run_episode()
            rewards.append(r)

        end = time()
        print "Test"
        print engine.get_actions_stats(clear=True, norm=False)
        m = np.mean(rewards)
        print "steps:", engine.get_steps(), ", mean:", m, "max:", np.max(rewards)
        if m > stop_mean:
            print stop_mean, "mean reached!"
            break
    epoch += 1
    print "========================="
overall_end = time()

print "Elapsed time:", overall_end - overall_start
