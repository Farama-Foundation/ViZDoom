from vizia import *
import numpy as np

from qengine import QEngine
from qengine import IdentityImageConverter
from evaluators import *
from time import time, sleep
import lasagne
import theano
import theano.tensor as T

from lasagne.regularization import l1, l2
from lasagne.updates import *
from lasagne.nonlinearities import leaky_rectify
from random import choice 
from transition_bank import TransitionBank
from lasagne.layers import get_all_param_values
from collections import OrderedDict

import cv2

def double_tanh(x):
    return 2*theano.tensor.tanh(x)

def create_scaled_tanh(scale):
    scale = np.float32(scale)
    def scaled_tanh(x):
        return scale*theano.tensor.tanh(x/scale)
    return scaled_tanh

def create_cutoff(threshold):
    threshold = np.float32(threshold)
    def cutoff(x):
        return T.maximum(T.minimum(threshold,x),-threshold)
    return cutoff

threshold = np.float32(0.0000001)
def clipped_sgd(regularized_loss, params, learning_rate):
    grads = lasagne.updates.get_or_compute_grads(regularized_loss, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        delta = learning_rate * grad
        delta = delta.clip(-threshold,threshold)
        updates[param] = param - delta
    return updates

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

def create_mlp_evaluator(state_format, actions_number, gamma):
    return None
def create_linear_evaluator(state_format, actions_number, gamma):
    eval_args = dict()
    eval_args["gamma"] = gamma
    eval_args["state_format"] = state_format
    eval_args["actions_number"] = actions_number
    eval_args["learning_rate"] = 0.01

    return LinearEvaluator(**eval_args)

def create_cnn_evaluator(state_format, actions_number, gamma):
    cnn_args = dict()
    cnn_args["gamma"] = gamma
    cnn_args["state_format"] = state_format
    cnn_args["actions_number"] = actions_number
    #cnn_args["updates"] = clipped_sgd
    cnn_args["learning_rate"] = 0.01
    #cnn_args["regularization"] = [[l1,1e-3],[l2,1e-3]]

    network_args = dict(hidden_units=[800], hidden_layers=1)
    network_args["conv_layers"] = 2
    network_args["pool_size"] = [(2, 2),(2,2),(1,2)]
    network_args["num_filters"] = [32,32,48]
    network_args["filter_size"] = [7,4,2]
    network_args["output_nonlin"] = None
    #network_args["output_nonlin"] = create_scaled_tanh(1000)
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
    engine_args['skiprate'] = 3
    engine_args['image_converter'] = ChannelScaleConverter
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 100000

    return engine_args

def engine_setup_health( game ):

    engine_args = dict()
    engine_args["evaluator"] = create_cnn_evaluator
    engine_args["game"] = game
    engine_args['gamma'] = 0.999
    engine_args["reward_scale"] = 0.01
    engine_args['skiprate'] = 8

    engine_args['actions_generator'] = agenerator_left_right_move
    engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True
    engine_args["count_states"] = True
    engine_args["misc_scale"] = [0.01,1.0]
    engine_args["epsilon_decay_steps"]=5000000
    engine_args["epsilon_decay_start_step"]=500000
    engine_args["batch_size"] = 100
    return engine_args