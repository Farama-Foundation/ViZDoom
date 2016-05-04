#!/usr/bin/python

#####################################################################
# This script presents how to make use of game variables to implement
# shaping using health_guided.wad scenario
# Health_guided scenario is just like health_gathering 
# (see "../../scenarios/README.md") but for each collected medkit global
# variable number 1 in acs script (coresponding to USER1) is increased
# by 100.0. It is not considered a part of reward but will possibly
# reduce learning time.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# 
#####################################################################
from __future__ import print_function
from vizdoom import *
from random import choice
import itertools as it
from time import sleep

import cv2

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("../../examples/config/health_gathering.cfg")
game.set_screen_resolution(ScreenResolution.RES_640X480)

game.init()

# Creates all possible actions.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))


episodes = 10
sleep_time = 0.028
last_total_shaping_reward = 0

for i in range(episodes):

    print("Episode #" + str(i+1))
    # Not needed for the first episdoe but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():


        # Gets the state and possibly to something with it
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        # Makes a random action and save the reward.
        r = game.make_action(choice(actions))

        # Retrieve the shaping reward
        sr = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
        sr = sr - last_total_shaping_reward
        last_total_shaping_reward += sr

        print("State #" +str(s.number))
        print("Health:", misc[0])
        print("Last Reward:", r)
        print("Last Shaping Reward:", sr)
        print("=====================")

        # Sleep some time because processing is too fast to watch.
        if sleep_time>0:
            sleep(sleep_time)

    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")

game.close()

