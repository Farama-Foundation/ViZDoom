#!/usr/bin/env python3

from __future__ import print_function

from random import choice
import vizdoom as vzd
from argparse import ArgumentParser
import cv2

DEFAULT_CONFIG = "../../scenarios/defend_the_center.cfg"

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use the 'automap' (top-down view map).")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()
    game = vzd.DoomGame()

    game.load_config(args.config)
    game.set_render_hud(False)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables rendering of automap.
    game.set_automap_buffer_enabled(True)

    # All map's geometry and objects will be displayed.
    game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    # Disables game window (FPP view), we just want to see the automap.
    game.set_window_visible(False)

    # This CVAR can be used to make a map follow a player.
    game.add_game_args("+am_followplayer 1")

    # This CVAR controls scale of rendered map (higher valuer means bigger zoom).
    game.add_game_args("+viz_am_scale 10")

    # This CVAR shows the whole map centered (overrides am_followplayer and viz_am_scale).
    game.add_game_args("+viz_am_center 1")

    # Map's colors can be changed using CVARs, full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
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
            print("Player position X:", state.game_variables[0], "Y:", state.game_variables[1], "Z:",
                  state.game_variables[2])

            print("=====================")

        print("Episode finished!")

        print("************************")

    cv2.destroyAllWindows()
