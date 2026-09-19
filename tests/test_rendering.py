#!/usr/bin/env python3

# Tests for rendering options and the screen buffer.

import os

import numpy as np

import vizdoom as vzd


def test_screen_buffer_always_uses_flashed_palette():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_screen_flashes(False)

    try:
        game.init()

        # testblend sets the renderer's base blend independently of
        # render_screen_flashes. The copied buffer must still use that palette.
        game.send_game_command("testblend red 1")
        game.advance_action()
        flashed_buffer = game.get_state().screen_buffer
        assert np.all(flashed_buffer == [255, 0, 0])
    finally:
        game.close()


def _screen_buffer_after_health_bonus(render_screen_flashes):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "predict_position.cfg"))
    game.set_seed(7)
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_screen_flashes(render_screen_flashes)
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.MOVE_FORWARD)

    try:
        game.init()
        initial_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        game.send_game_command("summon HealthBonus")

        for _ in range(20):
            game.make_action([1])
            if game.get_game_variable(vzd.GameVariable.HEALTH) > initial_health:
                return np.copy(game.get_state().screen_buffer)

        raise AssertionError("HealthBonus was not picked up")
    finally:
        game.close()


def test_render_screen_flashes_disables_pickup_flash():
    flashed_buffer = _screen_buffer_after_health_bonus(True)
    unflashed_buffer = _screen_buffer_after_health_bonus(False)

    changed_pixels = np.any(flashed_buffer != unflashed_buffer, axis=2)
    assert np.mean(changed_pixels) > 0.9
