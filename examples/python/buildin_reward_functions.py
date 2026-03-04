#!/usr/bin/env python3

#####################################################################
# This script presents how to set simple rewards that are built-in into ViZDoom
# Note that more complex rewards specific to a scenario can be programmed using ACS scripting language
#
# To see the scenario description go to
# https://vizdoom.farama.org/main/environments/default/
# #####################################################################

import os
from argparse import ArgumentParser
from time import sleep

import vizdoom as vzd


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "freedoom2.cfg")

if __name__ == "__main__":
    parser = ArgumentParser(
        "ViZDoom example showing built-in reward  in SPECTATOR mode."
    )
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
    args = parser.parse_args()

    game = vzd.DoomGame()

    game.load_config(args.config)

    # Enables freelook (mouse look) in the engine
    game.add_game_args("+freelook 1")

    # Increate screen resolution for better experience.
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    # Ensure variables needed for telemetry are available in state.
    game.set_available_game_variables(
        [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.ITEMCOUNT,
            vzd.GameVariable.KILLCOUNT,
        ]
    )

    # Built-in rewards setters
    game.set_health_reward(2.0)
    game.set_armor_reward(2.0)
    game.set_item_reward(1.0)
    game.set_kill_reward(10.0)
    game.set_death_penalty(
        100
    )  # Because it's a penalty, the value will be subtracted from the total reward, so it should be given as a positive number
    # game.set_death_reward(-100)  # As an alternative to set_death_penalty, you can set a negative reward for death using set_death_reward
    game.set_living_reward(-1)
    game.set_map_exit_reward(100.0)

    # Other available built-in reward setters
    # game.set_death_reward(0.0)
    # game.set_secret_reward(0.0)
    # game.set_frag_reward(0.0)
    # game.set_hit_reward(0.0)
    # game.set_hit_taken_reward(0.0)
    # game.set_hit_taken_penalty(0.0)
    # game.set_damage_made_reward(0.0)
    # game.set_damage_taken_reward(0.0)
    # game.set_damage_taken_penalty(0.0)

    print("Configured rewards:")
    print(f"health_reward={game.get_health_reward()}")
    print(f"armor_reward={game.get_armor_reward()}")
    print(f"item_reward={game.get_item_reward()}")
    print(f"kill_reward={game.get_kill_reward()}")
    print(f"death_reward={game.get_death_reward()}")
    print(f"living_reward={game.get_living_reward()}")
    print(f"map_exit_reward={game.get_map_exit_reward()}")
    print("=====================")

    game.init()

    for episode in range(args.episodes):
        print(f"Episode #{episode + 1}")
        print("Play the game and watch reward changes in the one-line log.")

        game.new_episode()

        prev_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        prev_armor = game.get_game_variable(vzd.GameVariable.ARMOR)
        prev_items = int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
        prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

        while not game.is_episode_finished():
            state = game.get_state()
            assert state is not None

            game.advance_action()

            health = game.get_game_variable(vzd.GameVariable.HEALTH)
            armor = game.get_game_variable(vzd.GameVariable.ARMOR)
            items = int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
            kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            total_reward = game.get_total_reward()

            delta_health = int(health - prev_health)
            delta_armor = int(armor - prev_armor)
            delta_items = items - prev_items
            delta_kills = kills - prev_kills

            print(
                f"State #{state.number} | "
                f"Health: {int(health)}({delta_health:+d}) | "
                f"Armor: {int(armor)}({delta_armor:+d}) | "
                f"Items: {items}({delta_items:+d}) | "
                f"Kills: {kills}({delta_kills:+d}) | "
                f"Reward: {total_reward}({game.get_last_reward():+0.3f}) | "
                f"Action: {game.get_last_action()}"
            )

            prev_health = health
            prev_armor = armor
            prev_items = items
            prev_kills = kills

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(1.0)

    game.close()
