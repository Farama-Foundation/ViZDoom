#!/usr/bin/env python3

# This test can be run as Python script or via PyTest

import os
from copy import deepcopy
from itertools import product
from random import choice

import numpy as np
import psutil
import vizdoom as vzd


def test_get_state():
    print("Testing get_state() ...")

    episodes = 10000
    episode_timeout = 1000
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
    game.set_episode_timeout(episode_timeout)
    game.set_available_buttons(buttons)
    game.init()

    prev_mem = 0
    prev_len = 0
    mem_eta = 1
    for i in range(episodes):
        game.new_episode()

        states = []
        states_copies = []

        while not game.is_episode_finished():
            state = game.get_state()
            states.append(state.screen_buffer)
            states_copies.append(np.copy(state.screen_buffer))

            game.make_action(choice(actions), 4)

        # Compare states with their copies
        for s, s_copy in zip(states, states_copies):
            assert np.array_equal(s, s_copy)

        # Check memory
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024

        if i % 100 == 0:
            print(
                f"Memory, with {len(states)} states saved, after episode {i} / {episodes}: {mem} MB"
            )

        if prev_len < len(states):
            prev_mem = mem
            prev_len = len(states)
        elif prev_len == len(states):
            assert abs(prev_mem - mem) < mem_eta


if __name__ == "__main__":
    test_get_state()
