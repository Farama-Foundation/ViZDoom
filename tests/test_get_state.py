#!/usr/bin/env python3

# Tests for get_state() method and returned State object.
# This test can be run as Python script or via PyTest.

import os
import pickle
import random
from itertools import product

import numpy as np
import psutil

import vizdoom as vzd


def _test_get_state(
    num_iterations=10,
    num_states=20,
    mem_eta_mb=0,
    depth_buffer=False,
    labels_buffer=False,
    automap_buffer=False,
    objects_info=False,
    sectors_info=False,
    audio_buffer=False,
    seed=1993,
):
    print("Testing get_state() ...")

    random.seed(seed)

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

    game.set_depth_buffer_enabled(depth_buffer)
    game.set_labels_buffer_enabled(labels_buffer)
    game.set_automap_buffer_enabled(automap_buffer)
    game.set_objects_info_enabled(objects_info)
    game.set_sectors_info_enabled(sectors_info)
    game.set_audio_buffer_enabled(audio_buffer)

    buffers = ["screen_buffer"]
    if depth_buffer:
        buffers.append("depth_buffer")
    if labels_buffer:
        buffers.append("labels_buffer")
    if automap_buffer:
        buffers.append("automap_buffer")
    if audio_buffer:
        buffers.append("audio_buffer")
        # This fixes "BiquadFilter_setParams: Assertion `gain > 0.00001f' failed" issue
        # or "no audio in buffer" issue caused by a bug in OpenAL version 1.19.
        game.add_game_args("+snd_efx 0")

    game.init()

    prev_mem = 0
    prev_len = 0
    for i in range(num_iterations):

        states = []
        buffers_copies = []

        game.new_episode()
        for _ in range(num_states):
            if game.is_episode_finished():
                game.new_episode()

            state = game.get_state()
            states.append(state)
            copies = {}
            for b in buffers:
                copies[b] = np.copy(getattr(state, b))
            buffers_copies.append(copies)
            game.make_action(random.choice(actions), 4)

        assert len(states) == num_states
        assert len(buffers_copies) == num_states

        # Compare states with their copies - confirms that states don't mutate.
        # Check min and max values of buffers - confirms that buffers are not empty.
        min_vals = {b: np.inf for b in buffers}
        max_vals = {b: -np.inf for b in buffers}
        for s, bs_copy in zip(states, buffers_copies):
            for b in buffers:
                assert np.array_equal(
                    getattr(s, b), bs_copy[b]
                ), f"Buffer {b} is not equal with its copy"
                min_vals[b] = min(min_vals[b], np.min(bs_copy[b]))
                max_vals[b] = max(max_vals[b], np.max(bs_copy[b]))

        for b in buffers:
            assert (
                min_vals[b] != max_vals[b]
            ), f"Buffer {b} min: {min_vals[b]}, max: {max_vals[b]} are equal, buffer is empty"

        # Save and load states via pickle - confirms that states and all sub-objects (labels, lines, objects) are picklable.
        with open("tmp_states.pkl", "wb") as f:
            pickle.dump(states, f)

        with open("tmp_states.pkl", "rb") as f:
            pickled_states = pickle.load(f)

        # Compare loaded states with their copies - to confirm that pickling doesn't mutate states.
        for s, s_copy in zip(states, pickled_states):
            assert pickle.dumps(s) == pickle.dumps(
                s_copy
            ), "Pickled state is not equal with its original object after save and load"

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
                assert (
                    abs(prev_mem - mem) < mem_eta_mb
                ), f"Memory leak detected: with {len(states)} states saved, after episode {i + 1} / {num_iterations}: {mem} MB used, expected ~{prev_mem} +/- {mem_eta_mb} MB"


def test_get_state(num_iterations=10, num_states=20):
    _test_get_state(num_iterations=num_iterations, num_states=num_states, mem_eta_mb=0)
    _test_get_state(
        num_iterations=num_iterations,
        num_states=num_states,
        mem_eta_mb=0,
        depth_buffer=True,
        labels_buffer=True,
        automap_buffer=True,
        objects_info=True,
        sectors_info=True,
        audio_buffer=False,  # Turned off by default, because it fails on some systems without audio backend and OpenAL installed
    )


if __name__ == "__main__":
    test_get_state()
