#!/usr/bin/python

#####################################################################
# This script presents how to run some scenarios.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import DoomGame, ScreenResolution
from random import choice
import itertools as it
from time import sleep

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("../../examples/config/basic.cfg")
#game.load_config("../../examples/config/deadly_corridor.cfg")
#game.load_config("../../examples/config/deathmatch.cfg")
#game.load_config("../../examples/config/defend_the_center.cfg")
#game.load_config("../../examples/config/defend_the_line.cfg")
#game.load_config("../../examples/config/health_gathering.cfg")
#game.load_config("../../examples/config/my_way_home.cfg")
#game.load_config("../../examples/config/predict_position.cfg")
#game.load_config("../../examples/config/take_cover.cfg")

# Makes the screen bigger to see more details.
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

# Creates all possible actions depending on how many buttons there are.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))


episodes = 10
sleep_time = 0.028

for i in range(episodes):
    print("Episode #" +str(i+1))

    # Not needed for the first episdoe but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():

        # Gets the state and possibly to something with it
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        # Makes a random action and save the reward.
        r = game.make_action(choice(actions))

        # Makes a "prolonged" action and skip frames:
        # skiprate = 3
        # r = game.make_action(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # skiprate = 3
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        print("State #" +str(s.number))
        print("Game Variables:", misc)
        print("Performed action:",game.get_last_action())
        print("Last Reward:",r)
        print("=====================")

        # Sleep some time because processing is too fast to watch.
        if sleep_time>0:
            sleep(sleep_time)

    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
