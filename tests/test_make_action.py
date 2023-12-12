#!/usr/bin/env python3

# Tests for make_action method.
# This test can be run as Python script or via PyTest

import numpy as np

import vizdoom as vzd


def _init_game():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.set_available_buttons(
        [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
    )
    game.set_episode_start_time(35)
    game.init()

    return game


def _test_exception(func, error, msg):
    try:
        func()
    except error:
        pass
    else:
        assert False, msg


def _compare_actions(action_a, action_b):
    if isinstance(action_a, np.ndarray):
        assert np.equal(action_a, action_b).all()
    else:
        assert action_a == action_b


def _test_make_action_input(type_name, type_args={}):
    def _make_func(game, action, tics=1):
        return game.make_action(action, tics)

    def _set_advance_func(game, action, tics=1):
        game.set_action(action)
        game.advance_action(tics, True)
        return game.get_last_reward()

    __test_make_action_input(_make_func, type_name, type_args=type_args)
    __test_make_action_input(_set_advance_func, type_name, type_args=type_args)


def __test_make_action_input(
    make_action_func,
    type_name,
    type_args={},
):
    print(
        f"Testing make_action with {type_name.__name__}([], {type_args}) type as input ..."
    )

    # Prepare game
    game = _init_game()

    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)

    # make_action() with correct arguments
    next_action = type_name([1, 0, 1], **type_args)
    make_action_func(game, next_action, 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
    _compare_actions(next_action, type_name(game.get_last_action(), **type_args))

    # make_action() without skipping frames
    make_action_func(game, next_action)

    # make_action_func() with negative frames and other types
    error_msg = "make_action() should raise TypeError when called with negative frames or type other than unsigned int"
    _test_exception(
        lambda: make_action_func(game, next_action, -10), TypeError, error_msg
    )
    _test_exception(
        lambda: make_action_func(game, next_action, "10"), TypeError, error_msg
    )

    # make_action() with too short action
    next_action = type_name([1, 0], **type_args)
    make_action_func(game, next_action, 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    _compare_actions(next_action, type_name(game.get_last_action()[:2], **type_args))

    # make_action() with too long action
    next_action = type_name([1, 0, 1, 0], **type_args)
    make_action_func(game, next_action, 8)
    assert prev_pos_y < game.get_game_variable(vzd.GameVariable.POSITION_Y)
    prev_pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)
    prev_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
    _compare_actions(next_action[:3], type_name(game.get_last_action(), **type_args))

    # make_action() with values other than 0 and 1
    next_action = type_name([0, 5, -5], **type_args)
    make_action_func(game, next_action, 16)
    assert prev_pos_y > game.get_game_variable(vzd.GameVariable.POSITION_Y)
    assert prev_ammo > game.get_game_variable(vzd.GameVariable.AMMO2)
    _compare_actions(next_action[:3], type_name(game.get_last_action(), **type_args))


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

    _test_make_action_input(np.array, {"dtype": np.float16})
    _test_make_action_input(np.array, {"dtype": np.float32})
    _test_make_action_input(np.array, {"dtype": np.float64})
    _test_make_action_input(np.array, {"dtype": np.int16})
    _test_make_action_input(np.array, {"dtype": np.int32})
    _test_make_action_input(np.array, {"dtype": np.int64})


def test_make_action_tuple():
    def typed_tuple(data, dtype=int):
        return tuple([dtype(d) for d in data])

    _test_make_action_input(typed_tuple, {"dtype": int})
    _test_make_action_input(typed_tuple, {"dtype": float})
    _test_make_action_input(typed_tuple, {"dtype": bool})


def test_make_action_mixed_list():
    def mixed_typed_list(data, dtypes):
        return [dtypes[i](d) for i, d in enumerate(data)]

    _test_make_action_input(mixed_typed_list, {"dtypes": [int, float, bool, int]})
    _test_make_action_input(mixed_typed_list, {"dtypes": [float, bool, int, float]})
    _test_make_action_input(mixed_typed_list, {"dtypes": [bool, int, float, bool]})


def test_keyword_arguments():
    game = _init_game()

    game.make_action(action=[1, 1, 1], tics=10)
    game.set_action(action=[1, 0, 1])
    game.advance_action(update_state=False)
    game.advance_action(tics=10, update_state=True)
    game.advance_action(tics=1)


def test_advance_action():
    game = _init_game()

    # advance_action() without set_action()
    game.advance_action()

    # advance_action() with update_state argument set to False
    prev_tic = game.get_state().tic
    tics_to_advance = 10
    game.advance_action(tics_to_advance, update_state=False)
    assert game.get_state().tic == prev_tic
    game.advance_action()
    assert game.get_state().tic == prev_tic + tics_to_advance + 1


if __name__ == "__main__":
    test_make_action_list()
    test_make_action_numpy()
    test_make_action_tuple()
    test_make_action_mixed_list()
    test_keyword_arguments()
    test_advance_action()
