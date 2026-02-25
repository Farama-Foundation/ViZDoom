#!/usr/bin/env python3

from random import choice
from time import time

from tqdm import tqdm

import vizdoom as vzd


DEFAULT_ITERATIONS = 10000
DEFAULT_INIT_CLOSE_ITERATIONS = 100


def _test_steps(game, steps=DEFAULT_ITERATIONS, skip=1):
    actions_num = game.get_available_buttons_size()
    actions = [
        [True if i == j else False for i in range(actions_num)]
        for j in range(actions_num)
    ]
    actions.append([False for _ in range(actions_num)])  # Idle action

    start = time()
    game.init()

    for _ in tqdm(range(steps)):
        if game.is_episode_finished():
            game.new_episode()

        # Copying happens here
        _ = game.get_state()
        if actions is not None:
            game.make_action(choice(actions), skip)
        else:
            game.advance_action(skip)

    end = time()
    t = end - start

    print("Results: Time:", round(t, 3), "s", "FPS:", round(steps / t, 2))


def test_screen_formats():
    print(
        "Testing steps performance with different screen formats. It may take some time. Be patient."
    )
    for screen_foramt in [
        vzd.ScreenFormat.CRCGCB,
        vzd.ScreenFormat.RGB24,
        vzd.ScreenFormat.GRAY8,
    ]:
        print(f"Testing with screen format: {screen_foramt}")
        g = vzd.DoomGame()
        g.set_screen_format(screen_foramt)
        g.set_window_visible(False)
        _test_steps(g)
        g.close()
        print("---------------------")
    print("=====================")


def test_buffers():
    buffers = [
        "depth_buffer",
        "labels_buffer",
        "automap_buffer",
        "audio_buffer",
        "objects_info",
        "sectors_info",
        "notifications_buffer",
    ]

    print(
        "Testing steps performance with different buffers enabled. It may take some time. Be patient."
    )
    for buffer in buffers:
        print(f"Testing with {buffer} enabled")
        g = vzd.DoomGame()
        g.set_window_visible(False)
        getattr(g, f"set_{buffer}_enabled")(True)
        _test_steps(g)
        g.close()
    print("=====================")

    print("Testing with all buffers enabled")
    g = vzd.DoomGame()
    g.set_window_visible(False)
    for buffer in buffers:
        getattr(g, f"set_{buffer}_enabled")(True)
    _test_steps(g)
    g.close()
    print("=====================")


def test_init_close(iterations=DEFAULT_INIT_CLOSE_ITERATIONS):
    print("Testing init/close performance. It may take some time. Be patient.")

    g = vzd.DoomGame()
    g.set_window_visible(False)
    start = time()
    for _ in tqdm(range(iterations)):
        g.init()
        g.close()
    end = time()
    t = end - start

    print(
        "Results: Time:",
        round(t, 3),
        "s",
        "Init/close per second:",
        round(iterations / t, 2),
        "Average time per init/close:",
        round(t / iterations, 3),
        "s",
    )
    print("=====================")


if __name__ == "__main__":
    test_screen_formats()
    test_buffers()
    test_init_close()
