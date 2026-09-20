"""Exercise the SDL video paths without requiring a desktop display."""

import sys

import numpy as np
import pytest

import vizdoom as vzd


@pytest.mark.skipif(sys.platform == "win32", reason="Windows uses native video")
@pytest.mark.parametrize("force_surface", [False, True])
@pytest.mark.parametrize("display_bits", [15, 16, 24, 30, 32])
def test_sdl_video(monkeypatch, force_surface, display_bits):
    monkeypatch.setenv("SDL_VIDEO_DRIVER", "dummy")
    monkeypatch.setenv("SDL_RENDER_DRIVER", "software")
    game = vzd.DoomGame()
    # Hidden windows normally bypass SDL entirely. The dummy driver lets us
    # exercise window creation, texture/surface updates and shutdown in CI.
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_screen_flashes(True)
    game.add_game_args(
        f"+vid_forcesurface {int(force_surface)} +vid_displaybits {display_bits}"
    )

    try:
        game.init()
        for fullscreen in [True, False]:
            game.send_game_command(f"fullscreen {int(fullscreen)}")
            game.send_game_command(f"vid_vsync {int(fullscreen)}")
            game.send_game_command("testblend red 1")
            game.advance_action(2)
            state = game.get_state()
            assert state is not None
            assert np.all(state.screen_buffer == [255, 0, 0])
    finally:
        game.close()
