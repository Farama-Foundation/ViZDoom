import os

import numpy as np
import pytest

import vizdoom as vzd


@pytest.mark.parametrize("backend", ["headless", "surface", "renderer"])
@pytest.mark.parametrize(
    "screen_format", [vzd.ScreenFormat.RGB24, vzd.ScreenFormat.GRAY8]
)
def test_sdl_rendering_and_restart(monkeypatch, backend, screen_format):
    # Headless operation must not require a working display driver. The other
    # cases exercise a hidden SDL window without needing a desktop in CI.
    headless = backend == "headless"
    monkeypatch.setenv("SDL_VIDEODRIVER", "unavailable" if headless else "dummy")
    monkeypatch.setenv("SDL_RENDER_DRIVER", "software")

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(not headless)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(screen_format)
    game.add_game_args(
        f"+viz_window_hidden 1 +vid_forcesurface {int(backend == 'surface')}"
    )

    for _ in range(2):
        try:
            game.init()
            game.make_action([0] * game.get_available_buttons_size(), 3)
            state = game.get_state()
            expected_shape = (
                (120, 160, 3) if screen_format == vzd.ScreenFormat.RGB24 else (120, 160)
            )
            assert state.screen_buffer.shape == expected_shape
            assert np.any(state.screen_buffer)
            assert game.get_episode_time() >= 3
        finally:
            game.close()
