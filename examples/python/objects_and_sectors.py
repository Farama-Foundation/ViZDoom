#!/usr/bin/env python3

#####################################################################
# This script presents how to access the information
# about the actors and objects present in the current scenario
# and map sectors (geometry).
#
# This information is not available if "+viz_nocheat" flag is enabled.
#####################################################################

import os
from argparse import ArgumentParser
from random import choice

import matplotlib.pyplot as plt

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "my_way_home.cfg")


if __name__ == "__main__":
    parser = ArgumentParser(
        "ViZDoom example showing how to use information about objects and map."
    )
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()
            game.make_action(choice(actions))

            print("State #" + str(state.number))
            print(
                "Player position: x:",
                state.game_variables[0],
                ", y:",
                state.game_variables[1],
                ", z:",
                state.game_variables[2],
            )
            print("Objects:")

            # Print information about objects present in the episode.
            for o in state.objects:
                print("Object id:", o.id, "object name:", o.name)
                print(
                    "Object position: x:",
                    o.position_x,
                    ", y:",
                    o.position_y,
                    ", z:",
                    o.position_z,
                )

                # Other available fields:
                # print("Object rotation angle", o.angle, "pitch:", o.pitch, "roll:", o.roll)
                # print("Object velocity x:", o.velocity_x, "y:", o.velocity_y, "z:", o.velocity_z)

                # Plot object on map
                if o.name == "DoomPlayer":
                    plt.plot(o.position_x, o.position_y, color="green", marker="o")
                else:
                    plt.plot(o.position_x, o.position_y, color="red", marker="o")

            print("=====================")

            print("Sectors:")

            # Print information about sectors.
            for s in state.sectors:
                print(
                    "Sector floor height:",
                    s.floor_height,
                    ", ceiling height:",
                    s.ceiling_height,
                )
                print(
                    "Sector lines:",
                    [
                        (line.x1, line.y1, line.x2, line.y2, line.is_blocking)
                        for line in s.lines
                    ],
                )

                # Plot sector on map
                for line in s.lines:
                    if line.is_blocking:
                        plt.plot(
                            [line.x1, line.x2],
                            [line.y1, line.y2],
                            color="black",
                            linewidth=2,
                        )

            # Show map
            plt.show()

        print("Episode finished!")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
