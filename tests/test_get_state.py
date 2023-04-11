#!/usr/bin/env python3

# Tests for get_state() method and returned State object.
# This test can be run as Python script or via PyTest.

import os
import pickle
from itertools import product
from random import choice

import numpy as np
import psutil

import vizdoom as vzd


def _test_get_state(
    num_iterations=10,
    num_states=20,
    mem_eta_mb=0,
    depthBuffer=False,
    labelsBuffer=False,
    automapBuffer=False,
    objectsInfo=False,
    sectorsInfo=False,
):
    print("Testing get_state() ...")

    buttons = [
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.ATTACK,
        vzd.Button.USE,
    ]
    actions = [list(i) for i in product([0, 1], repeat=len(buttons))]

    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.set_episode_timeout(num_states)
    game.set_available_buttons(buttons)

    game.set_depth_buffer_enabled(depthBuffer)
    game.set_labels_buffer_enabled(labelsBuffer)
    game.set_automap_buffer_enabled(automapBuffer)
    game.set_objects_info_enabled(objectsInfo)
    game.set_sectors_info_enabled(sectorsInfo)

    game.init()

    prev_mem = 0
    prev_len = 0
    for i in range(num_iterations):

        states = []
        screen_buffer_copies = []

        game.new_episode()
        for _ in range(num_states):
            if game.is_episode_finished():
                game.new_episode()

            state = game.get_state()
            states.append(state)
            screen_buffer_copies.append(np.copy(state.screen_buffer))

            game.make_action(choice(actions), 4)

        assert len(states) == num_states
        assert len(screen_buffer_copies) == num_states

        # Compare states with their copies - confirms that states don't mutate.
        for s, sb_copy in zip(states, screen_buffer_copies):
            assert np.array_equal(s.screen_buffer, sb_copy)

        # Save and load states via pickle - confirms that states and all sub-objects (labels, lines, objects) are picklable.
        with open("tmp_states.pkl", "wb") as f:
            pickle.dump(states, f)

        with open("tmp_states.pkl", "rb") as f:
            pickled_states = pickle.load(f)

        # Compare loaded states with their copies - to confirm that pickling doesn't mutate states.
        for s, s_copy in zip(states, pickled_states):
            assert pickle.dumps(s) == pickle.dumps(s_copy)

        del pickled_states
        os.remove("tmp_states.pkl")

        # Check memory for leaks
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024

        if (i + 1) % 10 == 0:
            print(
                f"Memory, with {len(states)} states saved, after episode {i + 1} / {num_iterations}: {mem} MB, expected ~{prev_mem} +/- {mem_eta_mb} MB"
            )

        if mem_eta_mb > 0:
            if prev_len < len(states):
                prev_mem = mem
                prev_len = len(states)
            elif prev_len == len(states):
                assert abs(prev_mem - mem) < mem_eta_mb


def test_get_state(num_iterations=10, num_states=20):
    _test_get_state(num_iterations=num_iterations, num_states=num_states, mem_eta_mb=0)
    _test_get_state(
        num_iterations=num_iterations,
        num_states=num_states,
        mem_eta_mb=0,
        depthBuffer=True,
        labelsBuffer=True,
        automapBuffer=True,
        objectsInfo=True,
        sectorsInfo=True,
    )


if __name__ == "__main__":
    test_get_state()
