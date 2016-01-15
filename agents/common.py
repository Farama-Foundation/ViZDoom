from vizia import DoomGame
from vizia import Button
from vizia import GameVariable
from vizia import ScreenFormat
from vizia import ScreenResolution
import numpy as np

from qengine import QEngine
from qengine import IdentityImageConverter
from evaluators import CNNEvaluator
from time import time, sleep
import lasagne

from lasagne.regularization import l1, l2
from lasagne.updates import nesterov_momentum, sgd
from lasagne.nonlinearities import leaky_rectify
from random import choice 
from transition_bank import TransitionBank
from lasagne.layers import get_all_param_values

import cv2


# Different scenarios  initializations
def basic(game):
    game.set_doom_file_path("../scenarios/basic.wad")

    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)

    game.set_episode_timeout(300)
    game.set_living_reward(-1)

def health_gathering(game):
    game.set_doom_file_path("../scenarios/health_gathering.wad")

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)

    game.set_episode_timeout(2100)
    game.set_living_reward(0.25)
    game.set_death_penalty(100)

    game.add_available_game_variable(GameVariable.HEALTH)

def health_guided(game):
    game.set_doom_file_path("../scenarios/health_guided.wad")

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)

    game.set_episode_timeout(2100)
    game.set_living_reward(0.125)
    game.set_death_penalty(100)

    game.add_available_game_variable(GameVariable.HEALTH)


# Common functions for learn.py and watch.py 

def agenerator_left_right_move(the_game):
    idle = [0,0,0]
    left = [1,0,0]
    right = [0,1,0]
    move = [0,0,1]
    move_left = [1,0,1]
    move_right = [0,0,1]
    return [left, right, move, move_left, move_right]

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
    network_args["num_filters"] = [32,32,48]
    network_args["filter_size"] = [7,4,2]
    #network_args["hidden_nonlin"] = None

    cnn_args["network_args"] = network_args
    return CNNEvaluator(**cnn_args)

def setup_vizia( scenario=basic, init=False):
    game = DoomGame()

    game.set_screen_resolution(ScreenResolution.RES_160X100)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_doom_iwad_path("../scenarios/doom2.wad")
        
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False);

    game.set_window_visible(False)
    scenario(game)

    if init:
        game.init()
    return game

class ChannelScaleConverter(IdentityImageConverter):
    def __init__(self, source):
        self._source = source
        self.x = 60
        self.y = int(self.x*3/4) 
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
    
def create_engine( game ):
    engine_args = dict()
    engine_args["history_length"] = 1
    engine_args["bank_capacity"] = 10000
    #engine_args["bank"] = TransitionBank( capacity=10000, rejection_range = [-0.02,0.5], rejection_probability=0.95)
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['start_epsilon'] = 0.95
    engine_args['end_epsilon'] = 0.0
    engine_args['epsilon_decay_start_step'] = 500000
    engine_args['epsilon_decay_steps'] = 5000000
    engine_args['update_frequency'] = (4,4) #every 4 steps, 4 updates each time
    engine_args['batch_size'] = 40
    engine_args['gamma'] = 0.99
    engine_args['skiprate'] = 8
    engine_args['reward_scale'] = 0.01
 
    #engine_args['actions_generator'] = agenerator_left_right_move
    engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True

    engine = QEngine(**engine_args)
    return engine


