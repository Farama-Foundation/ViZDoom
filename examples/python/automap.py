#!/usr/bin/env python

from __future__ import print_function

from random import choice
from vizdoom import *

import cv2

game = DoomGame()

# Use other config file if you wish.
# game.load_config("../../scenarios/basic.cfg")
# game.load_config("../../scenarios/simpler_basic.cfg")
# game.load_config("../../scenarios/rocket_basic.cfg")
# game.load_config("../../scenarios/deadly_corridor.cfg")
# game.load_config("../../scenarios/deathmatch.cfg")
game.load_config("../../scenarios/defend_the_center.cfg")
# game.load_config("../../scenarios/defend_the_line.cfg")
# game.load_config("../../scenarios/health_gathering.cfg")
# game.load_config("../../scenarios/my_way_home.cfg")
# game.load_config("../../scenarios/predict_position.cfg")
# game.load_config("../../scenarios/take_cover.cfg")
game.set_render_hud(False)

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Set cv2 friendly format.
game.set_screen_format(ScreenFormat.BGR24)

# Enables automap rendering
game.set_automap_buffer_enabled(True)

# All map geometry and objects will be displayed
game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)

game.add_available_game_variable(GameVariable.POSITION_X)
game.add_available_game_variable(GameVariable.POSITION_Y)
game.add_available_game_variable(GameVariable.POSITION_Z)

# We just want to see a automap
game.set_window_visible(False)

# This CVAR can be used to make a map follow a player
game.add_game_args("+am_followplayer 1")

# This CVAR controls scale of rendered map (higher valuer means bigger zoom)
game.add_game_args("+viz_am_scale 10")

# This CVAR shows whole map centered (overrides am_followplayer and viz_am_scale)
game.add_game_args("+viz_am_center 1")

# Map's colors can be changed using CVARs, full list available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
game.add_game_args("+am_backcolor 000000")
game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

episodes = 10

# Sleep time between actions in ms
sleep_time = 28

for i in range(episodes):
    print("Episode #" + str(i + 1))
    seen_in_this_episode = set()

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():
        # Gets the state
        state = game.get_state()

        # Shows automap buffer
        map = state.automap_buffer
        if map is not None:
            cv2.imshow('ViZDoom Automap Buffer', map)

        cv2.waitKey(sleep_time)

        game.make_action(choice(actions))

        print("State #" + str(state.number))
        print("Player position X:", state.game_variables[0], "Y:", state.game_variables[1], "Z:", state.game_variables[2])

        print("=====================")

    print("Episode finished!")

    print("************************")

cv2.destroyAllWindows()