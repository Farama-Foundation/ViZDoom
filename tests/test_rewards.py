#!/usr/bin/env python3

import os

import pytest

import vizdoom as vzd


@pytest.mark.parametrize("multiplayer", [False, True])
@pytest.mark.parametrize("penalty", [10, -10])
def test_death_penalty_once_per_death(multiplayer, penalty):
    game = vzd.DoomGame()
    scenario = "multi.cfg" if multiplayer else "basic.cfg"
    game.load_config(os.path.join(vzd.scenarios_path, scenario))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_available_buttons([])
    game.set_living_reward(0)
    game.set_death_penalty(penalty)
    if multiplayer:
        game.add_game_args("-host 1 -deathmatch +sv_cheats 1")
    game.init()

    try:
        for _ in range(2):
            assert game.get_last_reward() == 0
            assert game.get_total_reward() == 0
            game.send_game_command("kill")
            assert game.make_action([]) == -penalty
            assert game.is_player_dead()
            assert game.get_total_reward() == -penalty

            if multiplayer:
                for tics in (1, 4, 8):
                    assert game.make_action([], tics) == 0
                    assert game.is_player_dead()
                    assert game.get_total_reward() == -penalty

                game.respawn_player()
                assert game.get_last_reward() == 0
                assert game.get_total_reward() == -penalty
                game.send_game_command("kill")
                assert game.make_action([]) == -penalty
                assert game.get_total_reward() == -2 * penalty

            game.new_episode()
    finally:
        game.close()


@pytest.mark.parametrize("deaths", [1, 2])
def test_death_penalty_with_automatic_respawn(deaths):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_available_buttons([])
    game.set_living_reward(0)
    game.set_death_penalty(10)
    game.add_game_args(
        "-host 1 -deathmatch +sv_cheats 1 +sv_forcerespawn 1 +viz_respawn_delay 0"
    )
    game.init()

    try:
        for _ in range(deaths):
            game.send_game_command("kill")
            game.advance_action(70, update_state=False)
            assert not game.is_player_dead()
        assert game.get_game_variable(vzd.GameVariable.DEATHCOUNT) == deaths

        assert game.make_action([]) == -10 * deaths
        assert game.get_total_reward() == -10 * deaths
        assert game.make_action([], 4) == 0
        assert game.get_total_reward() == -10 * deaths
    finally:
        game.close()
