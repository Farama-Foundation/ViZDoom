#!/usr/bin/env python3

import vizdoom as vzd

def _test_enums(enum_name, func_name):
    print("Testing vzd.{} enum ...".format(enum_name))

    game = vzd.DoomGame()
    game.set_window_visible(False)

    add_func = eval("game.add_" + func_name)
    get_func = eval("game.get_" + func_name + "s")
    set_func = eval("game.set_" + func_name + "s")
    clear_func = eval("game.clear_" + func_name + "s")

    all_values = [eval("vzd." + enum_name + "." + v) for v in dir(eval("vzd." + enum_name)) if not v.startswith('__') and not v == 'name' and not v == 'value']
    all_values_names = [v.name for v in all_values]

    # set test
    set_func(all_values)
    get_buttons_names = [v.name for v in get_func()]
    assert all_values_names == get_buttons_names

    # add test
    clear_func()
    for i, v in enumerate(all_values):
        add_func(v)
        get_values_names = [v.name for v in get_func()]
        assert all_values_names[:i + 1] == get_values_names

    # again set test
    set_func(all_values)
    get_values_names = [v.name for v in get_func()]
    assert all_values_names == get_values_names

    # multiple adds
    for i, v in enumerate(all_values):
        add_func(v)
        get_values_names = [v.name for v in get_func()]
        assert all_values_names == get_values_names

    # multiple in set
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
