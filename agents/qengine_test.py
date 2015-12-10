#!/usr/bin/python

import numpy as np
from qengine import QEngine
from games import ShootingDotGame
from evaluators import MLPEvaluator
from evaluators import CNNEvaluator
from evaluators import LinearEvaluator
from time import time, sleep
import itertools as it
import lasagne
from vizia import DoomGame
from vizia import Button
from vizia import GameVar
from lasagne.regularization import l1, l2
from lasagne.updates import adagrad, nesterov_momentum, sgd
from random import choice 
from transition_bank import TransitionBank
from lasagne.layers import get_all_param_values

def api_init_wrapper(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo):
    api.init(x, y, random_background, max_moves, living_reward, miss_penalty, hit_reward, ammo)

def actions_generator(the_game):
    n = the_game.get_action_format()
    actions = []
    for perm in it.product([False, True], repeat=n):
        actions.append(list(perm))
    return actions

def create_mlp_evaluator(state_format, actions_number, batch_size, gamma):
    mlp_args = dict()
    mlp_args["gamma"] = gamma
    mlp_args["state_format"] = state_format
    mlp_args["actions_number"] = actions_number
    mlp_args["batch_size"] = batch_size
    mlp_args["learning_rate"] = 0.01
    mlp_args["updates"] = nesterov_momentum
    #mlp_args["regularization"] = [[l1,0.001]]

    network_args = dict(hidden_units=[3000], hidden_layers=1)
    #network_args["hidden_nonlin"] = None
    #network_args["output_nonlin"] = None
    mlp_args["network_args"] = network_args

    return MLPEvaluator(**mlp_args)

def create_cnn_evaluator(state_format, actions_number, batch_size, gamma):
    cnn_args = dict()
    cnn_args["gamma"] = gamma
    cnn_args["state_format"] = state_format
    cnn_args["actions_number"] = actions_number
    cnn_args["batch_size"] = batch_size
    cnn_args["updates"] = lasagne.updates.nesterov_momentum
    #cnn_args["learning_rate"] = 0.01

    network_args = dict(hidden_units=[800], hidden_layers=1)
    network_args["conv_layers"] = 2
    network_args["pool_size"] = [(2, 2),(2,2),(1,2)]
    network_args["num_filters"] = [32,32,32]
    network_args["filter_size"] = [7,4,2]
    #network_args["hidden_nonlin"] = None
    #network_args["output_nonlin"] = None

    cnn_args["network_args"] = network_args
    return CNNEvaluator(**cnn_args)

def create_linear_evaluator(state_format,actions_number, batch_size, gamma):
    lin_args = dict()
    lin_args["gamma"] = gamma
    lin_args["state_format"] = state_format
    lin_args["actions_number"] = actions_number
    lin_args["batch_size"] = batch_size
    lin_args["learning_rate"] = 0.01
    lin_args["updates"] = lasagne.updates.nesterov_momentum
    network_args = dict()
    lin_args["network_args"] = network_args

    return LinearEvaluator(**lin_args)

def create_game():
    game_args = dict()
    game_args['x'] = 61
    game_args['y'] = 41
    
    game_args['hit_reward'] = 1.01
    game_args['max_moves'] = 300
    # should be positive cause it's treated as a penalty
    game_args['miss_penalty'] = 0.05
    # should be negative cause it's treated as a reward
    game_args['living_reward'] = -0.01
    game_args['random_background'] = True
    game_args['noise_level'] = 0.05
    game_args['ammo'] = np.inf

    game = ShootingDotGame(**game_args)
    return game

doom_actions = [[True,False,False],[False,True,False],[False,False,True]]
left = doom_actions[0]
right = doom_actions[1]
shoot = doom_actions[2]
idle = [False,False,False]


def setup_vizia():
    game = DoomGame()

    #available resolutions: 40x30, 60x45, 80x60, 100x75, 120x90, 160x120, 200x150, 320x240, 640x480
    game.set_screen_resolution(40,0)

    game.set_doom_game_path("../bin/viziazdoom")
    game.set_doom_iwad_path("../scenarios/doom2.wad")
    game.set_doom_file_path("../scenarios/s1_b.wad")
    game.set_doom_map("map01")
    game.set_episode_timeout_in_doom_tics(200)

    game.set_living_reward(-1)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False);

    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)

    print "Initializin DOOM ..."
    game.init()
    print "DOOM initialized ..."
    return game

def create_engine( game, online_mode=False ):
    engine_args = dict()
    engine_args["history_length"] = 1
    engine_args["bank_capacity"] = 10000
    #engine_args["bank"] = TransitionBank( capacity=10000, rejection_range = [-0.02,0.5], rejection_probability=0.95)
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['start_epsilon'] = 0.9
    engine_args['end_epsilon'] = 0.1
    engine_args['epsilon_decay_start_step'] = 1200000
    engine_args['epsilon_decay_steps'] = 20000000
    engine_args['actions_generator'] = actions_generator
    engine_args['update_frequency'] = (4,4)
    engine_args['batch_size'] = 40
    engine_args['gamma'] = 0.85
    engine_args['reward_scale'] = 0.01
    if online_mode:
        engine.online_mode = True
    engine = QEngine(**engine_args)
    return engine

game = setup_vizia()
engine = create_engine(game)

 
print "\nCreated network params:"
for p in get_all_param_values(engine.get_network()):
	print p.shape


epochs = np.inf
training_episodes_per_epoch = 200
test_episodes_per_epoch = 50
test_frequency = 1;
stop_mean = 1.0  # game.average_best_result()
overall_start = time()
print "Learning..."
epoch = 0
while epoch < epochs:
    engine.learning_mode = True
    rewards = []
    start = time()
    print "\nEpoch", epoch
    
    for episode in range(training_episodes_per_epoch):
        #if (episode+1)% (training_episodes_per_epoch/20)==0:
        #    print(episode+1)
        r = engine.run_episode()
        rewards.append(r)
        
    end = time()
    
    print "Train:"
    print engine.get_actions_stats(True)
    mean_loss = engine._evaluator.get_mean_loss()
    print "steps:", engine.get_steps(), ", mean:", np.mean(rewards), ", max:", np.max(
        rewards),"mean_loss:",mean_loss, "eps:", engine.get_epsilon()
    print "t:", round(end - start, 2)
    # learning mode off

    if (epoch+1) % test_frequency == 0 and test_episodes_per_epoch > 0:
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
        print "t:", round(end - start, 2)
    epoch += 1
    print "========================="
overall_end = time()

print "Elapsed time:", overall_end - overall_start
