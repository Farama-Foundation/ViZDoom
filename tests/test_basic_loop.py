#!/usr/bin/env python3

# Tests if basic game loop works correctly in different modes.
# This test can be run as Python script or via PyTest.

import vizdoom as vzd


def _test_basic_loop(mode, episodes=3, steps=10, frame_skip=1):
    game = vzd.DoomGame()
    game.set_mode(mode)
    game.set_window_visible(False)
    game.set_available_buttons(
        [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
    )
    game.set_episode_start_time(35)
    game.init()

    # Just run a few steps to see if anything crashes
    for _ in range(episodes):
        game.new_episode()
        for _ in range(steps):
            if game.is_episode_finished():
                break

            if mode in {vzd.Mode.ASYNC_SPECTATOR, vzd.Mode.SPECTATOR}:
                game.advance_action(frame_skip)
            else:
                game.make_action([0] * game.get_available_buttons_size(), frame_skip)

    game.close()


def test_basic_loop():
    modes = [
        vzd.Mode.PLAYER,
        vzd.Mode.ASYNC_PLAYER,
        vzd.Mode.SPECTATOR,
        vzd.Mode.ASYNC_SPECTATOR,
    ]

    frame_skips = [1, 4]

    for mode in modes:
        for frame_skip in frame_skips:
            print(f"Testing mode: {mode}, frame_skip: {frame_skip}")
            _test_basic_loop(mode, frame_skip=frame_skip)


if __name__ == "__main__":
    test_basic_loop()
