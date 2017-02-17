#!/usr/bin/env python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from time import sleep
from vizdoom import *

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../scenarios/basic.cfg")
# game.load_config("../../scenarios/simpler_basic.cfg")
# game.load_config("../../scenarios/rocket_basic.cfg")
# game.load_config("../../scenarios/deadly_corridor.cfg")
game.load_config("../../scenarios/deathmatch.cfg")
# game.load_config("../../scenarios/defend_the_center.cfg")
# game.load_config("../../scenarios/defend_the_line.cfg")
# game.load_config("../../scenarios/health_gathering.cfg")
# game.load_config("../../scenarios/my_way_home.cfg")
# game.load_config("../../scenarios/predict_position.cfg")
# game.load_config("../../scenarios/take_cover.cfg")


# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)

game.init()

episodes = 10

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        print("State #" + str(state.number))
        print("Game variables: ", state.game_variables)
        print("Action:", last_action)
        print("Reward:", reward)
        print("=====================")

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
