#!/usr/bin/python
#####################################################################
# This script presents different formats of the screen buffer.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../examples/config/basic.cfg" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.

# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import *
from time import sleep
from time import time
from random import choice
import cv2

game = DoomGame()

# Use other config file if you wish.
game.load_config("../../examples/config/basic.cfg")
#game.set_window_visible(False)

# Just umcomment desired format. The last uncommented will be applied.
# Formats with C were ommited cause they are not cv2 friendly

#game.set_screen_format(ScreenFormat.RGB24)
#game.set_screen_format(ScreenFormat.ARGB32)
#game.set_screen_format(ScreenFormat.GRAY8)

# This is most fun. It looks best if you inverse colors.
game.set_screen_format(ScreenFormat.DEPTH_BUFFER8)

#These formats can be use bet they do not make much sense for cv2, you'll just get mixed up colors.
#game.set_screen_format(ScreenFormat.BGR24)
#game.set_screen_format(ScreenFormat.RGBA32) 
#game.set_screen_format(ScreenFormat.BGRA32) 
#game.set_screen_format(ScreenFormat.ABGR32)

#This one makes no sense in particular
#game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

actions = [[True,False,False],[False,True,False],[False,False,True]]

episodes = 10
# sleep time in ms
sleep_time = 20

for i in range(episodes):
    print("Episode #" +str(i+1))
    # Not needed for the first episdoe but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():


        # Gets the state and possibly to something with it
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        # Gray8 shape is not cv2 compliant
        if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
            img = img.reshape(img.shape[1],img.shape[2],1)

        # Display the image here!
        cv2.imshow('Doom Buffer',img)
        cv2.waitKey(sleep_time)

        # Makes a random action and save the reward.
        r = game.make_action(choice(actions))

        print("State #" +str(s.number))
        print("Game Variables:", misc)
        print("Last Reward:",r)
        print("=====================")


    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")

cv2.destroyAllWindows()
