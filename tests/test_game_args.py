#!/usr/bin/env python3

# Tests for ViZDoom enums and related methods.
# This test can be run as Python script or via PyTest

import vizdoom as vzd


def test_game_args():
    print("Testing setting custom game arguments...")

    game = vzd.DoomGame()
    game.set_window_visible(False)

    args1 = "-deathmatch +timelimit 1 +sv_spawnfarthest 1"
    args2 = "+name ViZDoom +colorset 0"
    args_all = args1 + " " + args2

    game.set_game_args(args_all)
    assert game.get_game_args() == args_all, "Game args not set correctly."

    game.clear_game_args()
    game.add_game_args(args1)
    game.add_game_args(args2)
    assert game.get_game_args() == args_all, "Game args not set correctly."


if __name__ == "__main__":
    test_game_args()
