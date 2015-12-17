#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVar
from vizia import ScreenFormat
import numpy as np

from qengine import QEngine
from qengine import IdentityImageConverter
from mockvizia import MockDoomGame
from evaluators import MLPEvaluator
from evaluators import CNNEvaluator
from evaluators import LinearEvaluator
from time import time, sleep
import itertools as it
import lasagne

from lasagne.regularization import l1, l2
from lasagne.updates import adagrad, nesterov_momentum, sgd
from lasagne.nonlinearities import leaky_rectify
from random import choice 
from transition_bank import TransitionBank
from lasagne.layers import get_all_param_values
from theano.tensor import tanh

import cv2

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
    network_args["hidden_nonlin"] = leaky_rectify
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
    network_args["output_nonlin"] = double_tanh

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

def setup_mockvizia():
    game = MockDoomGame()
    game.set_screen_resolution(40,30)
    game.set_no_shooting_time(8)
    game.init()
    return game

def setup_vizia():
    game = DoomGame()

    #available resolutions: 40x30, 60x45, 80x60, 100x75, 120x90, 160x120, 200x150, 320x240, 640x480
    game.set_screen_resolution(120,90)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_doom_game_path("../bin/viziazdoom")
    game.set_doom_iwad_path("../scenarios/doom2.wad")
    game.set_doom_file_path("../scenarios/s1_b.wad")
    game.set_doom_map("map01")
    game.set_episode_timeout(300)

    game.set_living_reward(-2)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False);

    game.set_visible_window(False)

    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.set_action_interval(4)

    print "Initializin DOOM ..."
    game.init()
    print "\nDOOM initialized."
    return game

def double_tanh(x):
    return 2*tanh(x)

class BNWDisplayImageConverter(IdentityImageConverter):
    def __init__(self, source):
        self._source = source

    def convert(self, img):
        img =  np.float32(img)/255.0
       #if len(img.shape) == 3:
        #    bnw_img = np.ma.average(img,axis=0, weights=[0.2989,0.5870,0.1140])
        
        bnw_img = cv2.resize(img, (320, 240)) 
        cv2.imshow('image',bnw_img)
        cv2.waitKey(1)
        return img

class ScaleConverter(IdentityImageConverter):
    def __init__(self, source):
        self._source = source
        self.x = 60
        self.y = 45 
    def convert(self, img):

        img =  np.float32(img)/255.0
        img = cv2.resize(img[0], (self.x,self.y))
        img =  img.reshape(1,self.y,self.x)
        
        return img

    def get_screen_width(self):
        return self.x

    def get_screen_height(self):
        return self.y

class ChannelScaleConverter(IdentityImageConverter):
    def __init__(self, source):
        self._source = source
        self.x = 60
        self.y = 45 
    def convert(self, img):

        img =  np.float32(img)/255.0
        new_image = np.ndarray([img.shape[0], self.y, self.x], dtype=np.float32)
        for i in range(img.shape[0]):
            new_image[i] = cv2.resize( img[i], (self.x, self.y))
        return new_image

    def get_screen_width(self):
        return self.x

    def get_screen_height(self):
        return self.y
    

def create_engine( game, online_mode=False ):
    engine_args = dict()
    engine_args["history_length"] = 1
    engine_args["bank_capacity"] = 10000
    #engine_args["bank"] = TransitionBank( capacity=10000, rejection_range = [-0.02,0.5], rejection_probability=0.95)
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['start_epsilon'] = 0.9
    engine_args['end_epsilon'] = 0.1
    engine_args['epsilon_decay_start_step'] = 100000
    engine_args['epsilon_decay_steps'] = 100000
    engine_args['actions_generator'] = actions_generator
    engine_args['update_frequency'] = (4,4)
    engine_args['batch_size'] = 40
    engine_args['gamma'] = 0.99
    engine_args['reward_scale'] = 0.01
    
    #engine_args['image_converter'] = BNWDisplayImageConverter
    #engine_args['image_converter'] = ScaleConverter
    engine_args['image_converter'] = ChannelScaleConverter
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
training_episodes_per_epoch = 400
test_episodes_per_epoch = 100
test_frequency = 1;
stop_mean = 1.0  # game.average_best_result()
overall_start = time()


epoch = 0
print "\nLearning ..."
while epoch < epochs:
    engine.learning_mode = True
    rewards = []
    start = time()
    print "\nEpoch", epoch
    
    for episode in range(training_episodes_per_epoch):
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

    print ""
    engine.save_params("rgb_60_skip4.params")

    print "========================="


overall_end = time()

print "Elapsed time:", overall_end - overall_start
