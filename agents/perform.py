#!/usr/bin/python
from common import *
from vizia import *

game = DoomGame()
game.load_config("config_common.properties")
game.load_config("config_health_guided.properties")
#game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_60X45)
game.init()

steps = 500

def test_engine(eng, steps):
    print "==============================="
    print "Testing"
    all_start = time()
    for i in range(steps):
        
        if game.is_episode_finished():
            game.new_episode()
        eng.make_learning_step()
    start = time()
    for i in range(steps):
        eng.learn_batch()
    end = time()

    all_end = time()
    print "All:",round(all_end-all_start,3)
    print "Learning:",round(end-start,3)

def create_cnn_evaluator1(state_format, actions_number, gamma):
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
    network_args["output_nonlin"] = None
    #network_args["hidden_nonlin"] = None

    cnn_args["network_args"] = network_args
    return CNNEvaluator(**cnn_args)

def create_engine_one(game):
    # Engine one:
    engine_args = dict()
    engine_args["evaluator"] = create_cnn_evaluator1
    engine_args["game"] = game
    engine_args['gamma'] = 1
    engine_args['reward_scale'] = 0.01
    #engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True
    engine_args["count_states"] = True
    engine_args["update_pattern"]=[5*steps,0]
    engine_args["batch_size"] = 40
    engine_args['misc_scale'] = [100.0, 1/2100.0]
    #engine_args["history_length"] = 3
    return QEngine(**engine_args)

def create_cnn_evaluator2(state_format, actions_number, gamma):
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
    network_args["output_nonlin"] = None
    #network_args["hidden_nonlin"] = None

    cnn_args["network_args"] = network_args
    return CNNEvaluator(**cnn_args)
 
def create_engine_two(game):   
    # Engine two:
    engine_args = dict()
    engine_args["evaluator"] = create_cnn_evaluator2
    engine_args["game"] = game
    engine_args['gamma'] = 1
    engine_args['reward_scale'] = 0.01
    #engine_args['image_converter'] = ChannelScaleConverter
    engine_args["shaping_on"] = True
    #engine_args["count_states"] = True
    engine_args["update_pattern"]=[5*steps,0]
    engine_args["batch_size"] = 40
    #engine_args["history_length"] = 5
    return QEngine(**engine_args)

engine1 =  create_engine_one(game)
engine2 = create_engine_two(game)

disp_shit = False
if disp_shit:
    print "\nNetwork architecture:"
    for p in get_all_param_values(engine1.get_network()):
        print p.shape

    for p in get_all_param_values(engine2.get_network()):
        print p.shape

test_engine(engine1,steps)
#test_engine(engine2,steps)