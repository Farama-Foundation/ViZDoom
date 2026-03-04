#!/usr/bin/env python3

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
#
# To see the scenario description go to
# https://vizdoom.farama.org/main/environments/default/
#####################################################################

import os
from argparse import ArgumentParser
from time import sleep

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "deathmatch.cfg")

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use SPECTATOR mode.")
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        default=1,
        type=int,
        help="Number of episodes to play.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        default=1,
        type=int,
        help="Number of episodes to play.",
    )
    args = parser.parse_args()
    game = vzd.DoomGame()

    # Choose scenario config file.
    game.load_config(args.config)

    # Enables freelook (mouse look) in the engine.
    game.add_game_args("+freelook 1")

    # Increate screen resolution for better experience.
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)

    # Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    game.init()

    for i in range(args.episodes):
        print(f"Episode #{i + 1}")

        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            assert state is not None

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            print(
                f"State #{state.number} | "
                f"Game variables: {state.game_variables} | "
                f"Action: {last_action} | "
                f"Reward: {reward}"
            )

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

    game.close()
