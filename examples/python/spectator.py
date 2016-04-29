#!/usr/bin/python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import *
from time import sleep

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

#game.load_config("../../examples/config/basic.cfg")
#game.load_config("../../examples/config/deadly_corridor.cfg")
game.load_config("../../examples/config/deathmatch.cfg")
#game.load_config("../../examples/config/defend_the_center.cfg")
#game.load_config("../../examples/config/defend_the_line.cfg")
#game.load_config("../../examples/config/health_gathering.cfg")
#game.load_config("../../examples/config/my_way_home.cfg")
#game.load_config("../../examples/config/predict_position.cfg")
#game.load_config("../../examples/config/take_cover.cfg")

# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)

game.init()

episodes = 10
print("")
for i in range(episodes):
    print("Episode #" +str(i+1))

    game.new_episode()
    while not game.is_episode_finished():

        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        game.advance_action()
        a = game.get_last_action()
        r = game.get_last_reward()

        print("state #"+str(s.number))
        print("game variables: ", misc)
        print("action:", a)
        print("reward:",r)
        print("=====================")

    print("episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
