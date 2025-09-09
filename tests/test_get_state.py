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
    notifications_buffer=False,
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
    game.set_notifications_buffer_enabled(notifications_buffer)

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
        audio_buffer=True,
        notifications_buffer=True,
    )


def _check_state(
    state,
    expect_game_variables=False,
    expect_depth_buffer=False,
    expect_labels_buffer=False,
    expect_automap_buffer=False,
    expect_objects_info=False,
    expect_sectors_info=False,
    expect_audio_buffer=False,
    expect_notifications_buffer=False,
):
    assert state is not None
    assert isinstance(state.tic, int), f"State tic is: {state.tic}, expected int"
    assert isinstance(
        state.screen_buffer, np.ndarray
    ), f"Screen buffer is: {state.screen_buffer}, expected np.ndarray"

    if expect_game_variables:
        assert isinstance(
            state.game_variables, np.ndarray
        ), f"Game variables are: {state.game_variables}, expected np.ndarray"
    else:
        assert (
            state.game_variables is None
        ), f"Game variables are: {state.game_variables}, expected None"

    if expect_depth_buffer:
        assert isinstance(
            state.depth_buffer, np.ndarray
        ), f"Depth buffer is: {state.depth_buffer}, expected np.ndarray"
    else:
        assert (
            state.depth_buffer is None
        ), f"Depth buffer is: {state.depth_buffer}, expected None"

    if expect_labels_buffer:
        assert isinstance(
            state.labels_buffer, np.ndarray
        ), f"Labels buffer is: {state.labels_buffer}, expected np.ndarray"
    else:
        assert (
            state.labels_buffer is None
        ), f"Labels buffer is: {state.labels_buffer}, expected None"

    if expect_automap_buffer:
        assert isinstance(
            state.automap_buffer, np.ndarray
        ), f"Automap buffer is: {state.automap_buffer}, expected np.ndarray"
        assert state.screen_buffer.shape == state.automap_buffer.shape, (
            f"Screen buffer shape is: {state.screen_buffer.shape}, "
            f"automap buffer shape is: {state.automap_buffer.shape}, expected equal",
        )
    else:
        assert (
            state.automap_buffer is None
        ), f"Automap buffer is: {state.automap_buffer}, expected None"

    if expect_objects_info:
        assert isinstance(
            state.objects, list
        ), f"Objects are: {state.objects}, expected list"
    else:
        assert state.objects is None, f"Objects are: {state.objects}, expected None"

    if expect_sectors_info:
        assert isinstance(
            state.sectors, list
        ), f"Sectors are: {state.sectors}, expected list"
    else:
        assert state.sectors is None, f"Sectors are: {state.sectors}, expected None"

    if expect_audio_buffer:
        assert isinstance(
            state.audio_buffer, np.ndarray
        ), f"Audio buffer is: {state.audio_buffer}, expected np.ndarray"
    else:
        assert (
            state.audio_buffer is None
        ), f"Audio buffer is: {state.audio_buffer}, expected None"

    if expect_notifications_buffer:
        assert isinstance(
            state.notifications_buffer, str
        ), f"Notifications buffer is: {state.notifications_buffer}, expected str"
    else:
        assert (
            state.notifications_buffer is None
        ), f"Notifications buffer is: {state.notifications_buffer}, expected None"


def test_buffer_sizes():
    game = vzd.DoomGame()
    game.set_window_visible(False)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_audio_buffer_enabled(True)
    game.set_notifications_buffer_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
    game.set_audio_buffer_size(8)
    game.set_notifications_buffer_size(8)

    screen_width = game.get_screen_width()
    screen_height = game.get_screen_height()
    assert screen_width == 640
    assert screen_height == 480

    audio_buf_size = game.get_audio_buffer_size()
    notifications_buf_size = game.get_notifications_buffer_size()
    assert audio_buf_size == 8
    assert notifications_buf_size == 8

    game.init()
    state = game.get_state()
    _check_state(
        state,
        expect_depth_buffer=True,
        expect_labels_buffer=True,
        expect_automap_buffer=True,
        expect_audio_buffer=True,
        expect_notifications_buffer=True,
    )
    assert state.screen_buffer.shape == (3, screen_height, screen_width)  # type: ignore
    assert state.depth_buffer.shape == (screen_height, screen_width)  # type: ignore
    assert state.labels_buffer.shape == (screen_height, screen_width)  # type: ignore
    assert state.automap_buffer.shape == (3, screen_height, screen_width)  # type: ignore
    assert state.audio_buffer.shape == (44100 // 35 * audio_buf_size, 2)  # type: ignore
    assert isinstance(state.notifications_buffer, str)


def test_if_none():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.init()

    state = game.get_state()
    _check_state(state)


def test_types():
    game = vzd.DoomGame()
    game.set_window_visible(False)

    game.add_available_game_variable(vzd.GameVariable.AMMO2)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_audio_buffer_enabled(True)
    game.set_notifications_buffer_enabled(True)

    # This fixes "BiquadFilter_setParams: Assertion `gain > 0.00001f' failed" issue
    # or "no audio in buffer" issue caused by a bug in OpenAL version 1.19.
    game.add_game_args("+snd_efx 0")

    game.init()
    state = game.get_state()

    _check_state(
        state,
        expect_game_variables=True,
        expect_depth_buffer=True,
        expect_labels_buffer=True,
        expect_automap_buffer=True,
        expect_objects_info=True,
        expect_sectors_info=True,
        expect_audio_buffer=True,
        expect_notifications_buffer=True,
    )


def test_modifing_buffers_while_game_is_running():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_audio_buffer_size(4)
    game.set_notifications_buffer_size(4)

    # This fixes "BiquadFilter_setParams: Assertion `gain > 0.00001f' failed" issue
    # or "no audio in buffer" issue caused by a bug in OpenAL version 1.19.
    game.add_game_args("+snd_efx 0")

    screen_width = game.get_screen_width()
    screen_height = game.get_screen_height()
    audio_buf_size = game.get_audio_buffer_size()
    notifications_buf_size = game.get_notifications_buffer_size()

    game.init()

    # Initial state - no buffers enabled
    state = game.get_state()
    _check_state(state)

    game.advance_action(4)

    # Game is running, try to change buffers setting
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_audio_buffer_enabled(True)
    game.set_notifications_buffer_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_audio_buffer_size(8)
    game.set_notifications_buffer_size(8)

    game.advance_action(4)

    state = game.get_state()
    _check_state(state)

    assert not game.is_depth_buffer_enabled()
    assert not game.is_labels_buffer_enabled()
    assert not game.is_automap_buffer_enabled()
    assert not game.is_objects_info_enabled()
    assert not game.is_sectors_info_enabled()
    assert not game.is_audio_buffer_enabled()
    assert not game.is_notifications_buffer_enabled()
    assert game.get_screen_width() == screen_width == 320
    assert game.get_screen_height() == screen_height == 240
    assert game.get_audio_buffer_size() == audio_buf_size == 4
    assert game.get_notifications_buffer_size() == notifications_buf_size == 4

    game.close()
    game.init()

    state = game.get_state()
    _check_state(state)

    assert not game.is_depth_buffer_enabled()
    assert not game.is_labels_buffer_enabled()
    assert not game.is_automap_buffer_enabled()
    assert not game.is_objects_info_enabled()
    assert not game.is_sectors_info_enabled()
    assert not game.is_audio_buffer_enabled()
    assert not game.is_notifications_buffer_enabled()
    assert game.get_screen_width() == screen_width == 320
    assert game.get_screen_height() == screen_height == 240
    assert game.get_audio_buffer_size() == audio_buf_size == 4
    assert game.get_notifications_buffer_size() == notifications_buf_size == 4


def test_server_state():
    game = vzd.DoomGame()
    game.set_window_visible(False)
    game.init()

    state = game.get_state()
    server_state = game.get_server_state()

    assert server_state is not None
    assert (
        server_state.tic == state.tic
    ), f"State tic is: {state.tic}, server state tic is: {server_state.tic}, expected equal"
    assert (
        server_state.player_count == 1
    ), f"Server state player count is: {server_state.player_count}, expected 1 for singeplayer game"


# TODO: Add more tests for server state


if __name__ == "__main__":
    test_get_state()
    test_buffer_sizes()
    test_if_none()
    test_types()
    test_modifing_buffers_while_game_is_running()
    test_server_state()
