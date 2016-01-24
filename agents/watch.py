#!/usr/bin/python
from common import *
from vizia import ScreenResolution
#load_file = "params/health_guided_160to60_skip8"
loadfile = "params/health_guided_60_skip8_bcp"
game = DoomGame()
game.load_config("config_common.properties")
#game.load_config("config_health_guided.properties")
game.load_config("config_basic.properties")


game = DoomGame()
game.set_console_enabled(False)
game.load_config("config_common.properties")
game.load_config("config_health_guided.properties")
#game.load_config("config_basic.properties")
print "Initializing DOOM ..."
game.set_window_visible(True)
#game.set_screen_resolution(ScreenResolution.RES_320X240)
game.init()
print "\nDOOM initialized."

engine = QEngine.load(game, loadfile)

print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
	print p.shape

sleep_time = 0.05

episodes = 20
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        engine.make_step()

        s = game.get_state()
        img = s.image_buffer
        #print "HP:",s.game_variables
        
        if sleep_time>0:
        	sleep(sleep_time)
    print i+1,"Reward:", game.get_summary_reward()
game.close()
