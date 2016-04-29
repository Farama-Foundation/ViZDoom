#!/usr/bin/python

from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep
from time import time


game = DoomGame()

game.set_vizdoom_path("../../bin/vizdoom")

game.set_doom_game_path("../../scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.

game.set_doom_map("map01")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Adds delta buttons that will be allowed and set the maximum allowed value (optional).
game.add_available_button(Button.MOVE_FORWARD_BACKWARD_DELTA, 10)
game.add_available_button(Button.MOVE_LEFT_RIGHT_DELTA, 5)
game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA, 5)
game.add_available_button(Button.LOOK_UP_DOWN_DELTA)

# For normal buttons (binary) all values other than 0 are interpreted as pushed.
# For delta buttons values determine a precision/speed.
#
# For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
# of which the viewing angle will change.
#
# For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
# value is the speed of movement in a given direction (100 is close to the maximum speed).
action = [100, 10, 10, 1]

# If button's absolute value > max button's value then value = max value with original value sign.

# Delta buttons in spectator modes correspond to mouse movements.
# Maximum allowed values also apply to spectator modes.
# game.add_game_args("+freelook 1")    # Use this to enable looking around with the mouse.
# game.set_mode(Mode.SPECTATOR)

game.set_window_visible(True)

game.init()

episodes = 10
sleep_time = 0.028

for i in range(episodes):
    print("Episode #" + str(i+1))

    game.new_episode()

    while not game.is_episode_finished():

        s = game.get_state()
        r = game.make_action(action)

        t = game.get_episode_time()

        action[0] = t % 100 - 50
        action[1] = t % 100 - 50
        action[2] = t % 100 - 50
        if not t % 25:
            action[3] = -action[3]

        print("State #" + str(s.number))
        print("=====================")

        if sleep_time>0:
            sleep(sleep_time)

    print("Episode finished.")
    print("************************")

game.close()



