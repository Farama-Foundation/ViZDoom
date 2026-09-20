#!/usr/bin/env python3

# Tests if basic game loop works correctly in different modes.
# This test can be run as Python script or via PyTest.

import time

import vizdoom as vzd


WAIT_SECONDS = 10
TIC_TOLERANCE = 10


def _test_basic_loop(mode, episodes=3, steps=10, frame_skip=1):
    print(
        f"Testing basic loop with mode: {mode}, episodes: {episodes}, steps: {steps}, frame_skip: {frame_skip} ..."
    )
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
            _test_basic_loop(mode, frame_skip=frame_skip)


def test_async_mode_runs_in_real_time():
    print("Testing that async mode runs in real time ...")
    game = vzd.DoomGame()
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_window_visible(False)
    game.set_ticrate(vzd.DEFAULT_TICRATE)
    game.init()

    try:
        # Synchronize the state before measuring. In async mode, advance_action()
        # requests a state update without controlling when game tics happen.
        game.advance_action()
        state = game.get_state()
        start_tic = state.tic
        assert start_tic == game.get_episode_time()
        start_time = time.monotonic()

        time.sleep(WAIT_SECONDS)

        game.advance_action()
        state = game.get_state()
        elapsed_tics = state.tic - start_tic
        assert elapsed_tics == game.get_episode_time() - start_tic

        elapsed_seconds = time.monotonic() - start_time
        expected_tics = elapsed_seconds * vzd.DEFAULT_TICRATE

        assert (
            elapsed_tics >= WAIT_SECONDS * vzd.DEFAULT_TICRATE
        ), f"Elapsed tics {elapsed_tics} is less than expected {WAIT_SECONDS * vzd.DEFAULT_TICRATE}"
        assert (
            abs(elapsed_tics - expected_tics) <= TIC_TOLERANCE
        ), f"Elapsed tics {elapsed_tics} differ from expected {expected_tics} by more than {TIC_TOLERANCE}"
    finally:
        game.close()


def test_sync_mode_waits_for_advance_action():
    print("Testing that sync mode waits for make/advance_action() ...")
    game = vzd.DoomGame()
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.init()

    try:
        state = game.get_state()
        assert state.tic == 1
        assert game.get_episode_time() == 1

        game.advance_action()
        state = game.get_state()
        assert state.tic == 2
        assert game.get_episode_time() == 2

        time.sleep(WAIT_SECONDS)

        assert game.get_episode_time() == 2

        game.advance_action()
        state = game.get_state()
        assert state.tic == 3
        assert game.get_episode_time() == 3
    finally:
        game.close()


if __name__ == "__main__":
    test_basic_loop()
    test_async_mode_runs_in_real_time()
    test_sync_mode_waits_for_advance_action()
