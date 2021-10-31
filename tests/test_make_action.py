#!/usr/bin/env python3

import vizdoom as vzd
import numpy as np


def _test_make_action_input(type_name, type_args={}):
    print("Testing make_action with {}([], {}) type as input ...".format(type_name.__name__, type_args))

    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_episode_start_time(35)

    game.init()
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)

    game.make_action(type_name([1, 0, 1], **type_args), 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)

    # too short action
    game.make_action(type_name([1, 0], **type_args), 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)

    # too long action
    game.make_action(type_name([1, 0, 1, 0], **type_args), 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)

    # values other than 0 and 1
    game.make_action(type_name([0, 5, -5], **type_args), 16)
    assert prev_pos_y > game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)


def test_make_action_list():
    def typed_list(data, dtype=int):
        return [dtype(d) for d in data]

    _test_make_action_input(typed_list, {"dtype": int})
    _test_make_action_input(typed_list, {"dtype": float})
    _test_make_action_input(typed_list, {"dtype": bool})


def test_make_action_numpy():
    _test_make_action_input(np.array, {"dtype": int})
    _test_make_action_input(np.array, {"dtype": float})
    _test_make_action_input(np.array, {"dtype": bool})

    _test_make_action_input(np.array, {"dtype": np.float32})
    _test_make_action_input(np.array, {"dtype": np.float64})


def test_make_action_tuple():
    def typed_tuple(data, dtype=int):
        return tuple([dtype(d) for d in data])

    _test_make_action_input(typed_tuple, {"dtype": int})
    _test_make_action_input(typed_tuple, {"dtype": float})
    _test_make_action_input(typed_tuple, {"dtype": bool})


if __name__ == "__main__":
    test_make_action_list()
    test_make_action_numpy()
    test_make_action_tuple()
