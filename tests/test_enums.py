#!/usr/bin/env python3

# Tests for ViZDoom enums and related methods.
# This test can be run as Python script or via PyTest

import vizdoom as vzd


def _test_enums(enum_name, func_name):
    print(f"Testing vzd.{enum_name} enum ...")

    game = vzd.DoomGame()
    game.set_window_visible(False)

    add_func = eval("game.add_" + func_name)
    get_func = eval("game.get_" + func_name + "s")
    set_func = eval("game.set_" + func_name + "s")
    clear_func = eval("game.clear_" + func_name + "s")

    all_values = [
        eval("vzd." + enum_name + "." + v)
        for v in dir(eval("vzd." + enum_name))
        if not v.startswith("__") and not v == "name" and not v == "value"
    ]
    all_values_names = [v.name for v in all_values]

    # set_X function test
    set_func(all_values)
    get_values_names = [v.name for v in get_func()]
    assert all_values_names == get_values_names

    # add_X function test
    clear_func()
    for i, v in enumerate(all_values):
        add_func(v)
        get_values_names = [v.name for v in get_func()]
        assert all_values_names[: i + 1] == get_values_names

    # Check if set function overwrites previous values
    set_func(all_values)
    get_values_names = [v.name for v in get_func()]
    assert all_values_names == get_values_names

    # Multiple add_X functions test
    for i, v in enumerate(all_values):
        add_func(v)
        get_values_names = [v.name for v in get_func()]
        assert all_values_names == get_values_names

    # Test duplicated values in set_X function
    set_func(all_values + all_values)
    get_values_names = [v.name for v in get_func()]
    assert all_values_names == get_values_names


def test_buttons():
    _test_enums("Button", "available_button")


def test_game_variables():
    _test_enums("GameVariable", "available_game_variable")


if __name__ == "__main__":
    test_buttons()
    test_game_variables()
