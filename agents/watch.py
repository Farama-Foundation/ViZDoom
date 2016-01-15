#!/usr/bin/python
from common import *
from vizia import ScreenResolution
load_file = "params/basic_120to60"

game = setup_vizia(scenario=basic,init=False)
print "Initializing DOOM ..."
game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.init()
print "\nDOOM initialized."
engine = create_engine(game)

 
print "\nCreated network params:"
for p in get_all_param_values(engine.get_network()):
	print p.shape

engine.load_params(load_file)


episodes = 20

sleep_time = 0.03
for i in range(episodes):

    game.new_episode()
    while not game.is_episode_finished():
        engine.make_step()
        img = game.get_state().image_buffer
        sleep(sleep_time)
    print i+1,"Reward:", game.get_summary_reward()
print "Finished"
print "Doom hangs during close :(."
game.close()
