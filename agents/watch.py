#!/usr/bin/python
from common import *
from vizia import ScreenResolution
load_file = "params/health_guided_160to60_skip8"

game = DoomGame()
game.load_config("config_common.properties")
game.load_config("config_health_guided.properties")
#game.load_config("config_basic.properties")

print "Initializing DOOM ..."
game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.init()
print "\nDOOM initialized."

engine_args = engine_setup(game)
engine_args['image_converter'] = ChannelScaleConverter
#engine_args["skiprate"] = 1
engine = QEngine(**engine_args)


 
print "\nCreated network params:"
for p in get_all_param_values(engine.get_network()):
	print p.shape

engine.load_params(load_file)


episodes = 20

sleep_time = 0.1
for i in range(episodes):

    game.new_episode()
    while not game.is_episode_finished():
        engine.make_step()
        img = game.get_state().image_buffer
        sleep(sleep_time)
    print i+1,"Reward:", game.get_summary_reward()
game.close()
