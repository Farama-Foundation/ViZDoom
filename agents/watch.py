#!/usr/bin/python
from common import *
from vizia import ScreenResolution
load_file = "params/health_guided_160_to60_skip8_3l_f48"

game = setup_vizia(scenario=health_guided,init=False)
print "Initializing DOOM ..."
game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_120X90)
game.init()
print "\nDOOM initialized."

engine_args = engine_setup(game)
engine_args["skiprate"] = 1
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
