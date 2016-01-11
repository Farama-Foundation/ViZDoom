#!/usr/bin/python
from vizia import DoomGame
from vizia import Button
from vizia import GameVar
from vizia import ScreenFormat
from vizia import ScreenResolution
import numpy as np

from qengine import QEngine
from qengine import IdentityImageConverter
from evaluators import CNNEvaluator
from time import time, sleep
import itertools as it
import lasagne

from lasagne.regularization import l1, l2
from lasagne.updates import nesterov_momentum, sgd
from lasagne.nonlinearities import leaky_rectify
from random import choice 
from transition_bank import TransitionBank
from lasagne.layers import get_all_param_values
from theano.tensor import tanh

import cv2

savefile = None

#savefile = "params/basic_120to60"
savefile = "params/health_guided_120_to60_skip4"
#savefile = "params/center_120_to80_skip4"
#savefile = "params/s1b_120_to60_skip1"
loadfile = None

def double_tanh(x):
    return 2*tanh(x)

def actions_generator(the_game):
    n = the_game.get_available_buttons_size()
    actions = []
    for perm in it.product([False, True], repeat=n):
        actions.append(list(perm))
    return actions

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

    game.add_state_available_var(GameVar.HEALTH)

def health_guided(game):
    game.set_doom_file_path("../scenarios/health_guided.wad")

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)

    game.set_episode_timeout(2100)
    game.set_living_reward(0.25)
    game.set_death_penalty(100)

    game.add_state_available_var(GameVar.HEALTH)

def defend_the_center(game):
    game.set_doom_file_path("../scenarios/defend_the_center")

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)

    game.set_episode_timeout(2100)
    game.set_living_reward(0)
    game.set_death_penalty(1)

    game.add_state_available_var(GameVar.HEALTH)

def setup_vizia():
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
    
    #basic(game)
    #health_gathering(game)
    health_guided(game)
    #defend_the_center(game)

    print "Initializing DOOM ..."
    game.init()
    print "\nDOOM initialized."
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
    engine_args['actions_generator'] = actions_generator
    engine_args['update_frequency'] = (4,4) #every 4 steps, 4 updates each time
    engine_args['batch_size'] = 40
    engine_args['gamma'] = 0.99
    engine_args['skiprate'] = 4
    engine_args['reward_scale'] = 0.01
 
    engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True

    engine = QEngine(**engine_args)
    return engine

game = setup_vizia()
engine = create_engine(game)

if loadfile:
    engine.load_params(loadfile)

print "\nCreated network params:"
for p in get_all_param_values(engine.get_network()):
	print p.shape


epochs = np.inf
training_episodes_per_epoch = 400
test_episodes_per_epoch = 100
test_frequency = 1;
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
        print "t:", round(end - start, 2)
    epoch += 1

    print ""

    if savefile:
        engine.save_params(savefile)

    overall_end = time()
    print "Elapsed time:", round(overall_end - overall_start,2), "s"  
    print "========================="


overall_end = time()
print "Elapsed time:", round(overall_end - overall_start,2), "s"  
