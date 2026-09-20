#!/usr/bin/env python3

import os

import vizdoom as vzd


def test_weapon_related_variables():
    weapon_slots = (
        (
            ("Fist", "Chainsaw"),
            vzd.Button.SELECT_WEAPON1,
            vzd.GameVariable.AMMO1,
            vzd.GameVariable.WEAPON1,
        ),
        (
            ("Pistol",),
            vzd.Button.SELECT_WEAPON2,
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.WEAPON2,
        ),
        (
            ("Shotgun", "SuperShotgun"),
            vzd.Button.SELECT_WEAPON3,
            vzd.GameVariable.AMMO3,
            vzd.GameVariable.WEAPON3,
        ),
        (
            ("Chaingun",),
            vzd.Button.SELECT_WEAPON4,
            vzd.GameVariable.AMMO4,
            vzd.GameVariable.WEAPON4,
        ),
        (
            ("RocketLauncher",),
            vzd.Button.SELECT_WEAPON5,
            vzd.GameVariable.AMMO5,
            vzd.GameVariable.WEAPON5,
        ),
        (
            ("PlasmaRifle",),
            vzd.Button.SELECT_WEAPON6,
            vzd.GameVariable.AMMO6,
            vzd.GameVariable.WEAPON6,
        ),
        (
            ("BFG9000",),
            vzd.Button.SELECT_WEAPON7,
            vzd.GameVariable.AMMO7,
            vzd.GameVariable.WEAPON7,
        ),
    )
    ammo_variables = [ammo_variable for _, _, ammo_variable, _ in weapon_slots]
    weapon_variables = [weapon_variable for _, _, _, weapon_variable in weapon_slots]
    all_weapon_variables = (
        ammo_variables
        + weapon_variables
        + [vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.SELECTED_WEAPON]
    )

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "freedoom2.cfg"))
    game.set_window_visible(False)
    game.set_available_buttons([button for _, button, _, _ in weapon_slots])
    game.set_available_game_variables(all_weapon_variables)
    game.set_seed(0)
    game.init()

    try:
        for weapons, _, _, _ in weapon_slots:
            for weapon in weapons:
                game.send_game_command(f"give {weapon}")
        game.send_game_command("give ammo")

        no_action = [False] * game.get_available_buttons_size()
        game.make_action(no_action)
        for (weapons, _, _, _), weapon_variable in zip(weapon_slots, weapon_variables):
            assert game.get_game_variable(weapon_variable) == len(weapons)

        for slot, (_, select_button, ammo_variable, weapon_variable) in enumerate(
            weapon_slots, start=1
        ):
            action = no_action.copy()
            action[game.get_available_buttons().index(select_button)] = True
            game.make_action(action)
            game.make_action(no_action)
            state = game.get_state()
            game_variable = state.game_variables

            # weapon checks
            selected_weapon = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON)
            assert selected_weapon == slot
            assert (
                game_variable[
                    all_weapon_variables.index(vzd.GameVariable.SELECTED_WEAPON)
                ]
                == selected_weapon
            )
            assert game_variable[all_weapon_variables.index(weapon_variable)] >= 1

            # ammo checks
            selected_weapon_ammo = game.get_game_variable(
                vzd.GameVariable.SELECTED_WEAPON_AMMO
            )
            assert (
                game_variable[
                    all_weapon_variables.index(vzd.GameVariable.SELECTED_WEAPON_AMMO)
                ]
                == selected_weapon_ammo
            )
            assert (
                game_variable[all_weapon_variables.index(ammo_variable)]
                == selected_weapon_ammo
            )
            assert game.get_game_variable(ammo_variable) == selected_weapon_ammo

    finally:
        game.close()


if __name__ == "__main__":
    test_weapon_related_variables()
