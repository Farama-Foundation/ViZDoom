from vizia import *
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

def sec_to_str(sec):
    res = str(int(sec%60)) +"s"
    sec = int(sec/60)
    if sec>0:
        res = str(int(sec%60)) +"m "+res
        sec = int(sec/60)
        if sec>0:
            res = str(int(sec%60)) +"h " +res
            sec = int(sec/60)
    return res
# Common functions for learn.py and watch.py 
def agenerator_left_right_move(the_game):
    idle = [0,0,0]
    left = [1,0,0]
    right = [0,1,0]
    move = [0,0,1]
    move_left = [1,0,1]
    move_right = [0,1,1]
    return [idle,left, right, move]

def create_cnn_evaluator(state_format, actions_number, gamma):
    cnn_args = dict()
    cnn_args["gamma"] = gamma
    cnn_args["state_format"] = state_format
    cnn_args["actions_number"] = actions_number
    cnn_args["updates"] = lasagne.updates.nesterov_momentum
    #cnn_args["learning_rate"] = 0.01

    network_args = dict(hidden_units=[800], hidden_layers=1)
    network_args["conv_layers"] = 2
    network_args["pool_size"] = [(2, 2),(2,2),(1,2)]
    network_args["num_filters"] = [32,32,48]
    network_args["filter_size"] = [7,4,2]
    #network_args["output_nonlin"] = None
    #network_args["hidden_nonlin"] = None

    cnn_args["network_args"] = network_args
    return CNNEvaluator(**cnn_args)


reshape_x = 60
class ChannelScaleConverter(IdentityImageConverter):
    def __init__(self, source):
        self._source = source
        self.x = reshape_x
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
    
def engine_setup_basic( game ):
    engine_args = dict()
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['gamma'] = 0.99
    engine_args["reward_scale"] = 0.01
    engine_args['skiprate'] = 8
    #engine_args['image_converter'] = ChannelScaleConverter

    return engine_args

def engine_setup_health( game ):

    engine_args = dict()
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['gamma'] = 1
    engine_args["reward_scale"] = 0.01
    engine_args['skiprate'] = 8

    engine_args['actions_generator'] = agenerator_left_right_move
    #engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True
    engine_args["count_states"] = True
    engine_args["misc_scale"] = [100.0, 1/2100.0]
    return engine_args