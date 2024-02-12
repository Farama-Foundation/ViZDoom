#!/usr/bin/env python3ch

# Tests ViZDoom seed option.
# This test can be run as Python script or via PyTest

import itertools
import os
import random

import cv2
import numpy as np

import vizdoom as vzd


def test_seed(repeats=10, tics=8, audio_buffer=False, seed=1993):
    scenarios_to_skip = [
        # "deadly_corridor.cfg",
        # "defend_the_center.cfg",
        # "deathmatch.cfg",
        # "health_gathering.cfg",
        # "health_gathering_supreme.cfg",
        # "deathmatch.cfg",
        # Multiplayer scenarios
        "cig.cfg",
        "multi_duel.cfg",
        "multi.cfg",
        "oblige.cfg",
    ]
    configs = [
        file
        for file in os.listdir(vzd.scenarios_path)
        if file.endswith(".cfg") and file not in scenarios_to_skip
    ]
    print(configs)
    game = vzd.DoomGame()

    for config in configs:
        print(config)
        initial_states = []
        states_after_action = []

        game = vzd.DoomGame()
        game.load_config(config)
        game.set_window_visible(False)

        # Creates all possible actions depending on how many buttons there are.
        actions_num = game.get_available_buttons_size()
        actions = []
        for perm in itertools.product([False, True], repeat=actions_num):
            actions.append(list(perm))

        # Enable all buffers
        buffers = ["screen_buffer", "depth_buffer", "labels_buffer", "automap_buffer"]
        game.set_depth_buffer_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_automap_buffer_enabled(True)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_audio_buffer_enabled(audio_buffer)
        if audio_buffer:
            buffers.append("audio_buffer")

        game.set_screen_format(vzd.ScreenFormat.BGR24)

        game.init()

        for i in range(repeats):
            game.set_seed(1993)
            random.seed(seed)
            # game.init()
            game.new_episode()

            initial_states.append(game.get_state())
            if i % 2 == 0:
                game.make_action(random.choice(actions), tics=tics)
            else:
                action = random.choice(actions)
                for _ in range(tics):
                    game.make_action(action, tics=1)

            game.make_action(random.choice(actions), tics=tics)
            states_after_action.append(game.get_state())

            # game.close()

        for s1, s2 in zip(initial_states[:-1], initial_states[1:]):
            assert s1.tic == s2.tic
            assert np.array_equal(s1.game_variables, s2.game_variables)

            if not np.array_equal(s1.screen_buffer, s2.screen_buffer):
                print("Initial states are not equal")
                print(f"s1: {s1.tic}, {s1.game_variables}")
                print(f"s2: {s2.tic}, {s2.game_variables}")
                print(np.all(s1.screen_buffer == s2.screen_buffer))
                print(np.array_equal(s1.screen_buffer, s2.screen_buffer))
                cv2.imshow("s1", s1.screen_buffer)
                cv2.imshow("s2", s2.screen_buffer)
                cv2.imshow("s1 - s2", s1.screen_buffer - s2.screen_buffer)
                cv2.waitKey(int(10000))

            for b in buffers:
                if not np.array_equal(getattr(s1, b), getattr(s2, b)):
                    print("Initial states are not equal")
                    cv2.imshow("s1", getattr(s1, b))
                    cv2.imshow("s2", getattr(s2, b))
                    cv2.imshow("s1 - s2", getattr(s1, b) - getattr(s2, b))
                    cv2.waitKey(int(10000))

                # assert np.array_equal(getattr(s1, b), getattr(s2, b))

        for s1, s2 in zip(states_after_action[:-1], states_after_action[1:]):
            assert s1.tic == s2.tic
            assert np.array_equal(s1.game_variables, s2.game_variables)

            if not np.array_equal(s1.screen_buffer, s2.screen_buffer):
                print("States after action are not equal")
                print(f"s1: {s1.tic}, {s1.game_variables}")
                print(f"s2: {s2.tic}, {s2.game_variables}")
                print(np.all(s1.screen_buffer == s2.screen_buffer))
                print(np.array_equal(s1.screen_buffer, s2.screen_buffer))
                cv2.imshow("s1", s1.screen_buffer)
                cv2.imshow("s2", s2.screen_buffer)
                cv2.imshow("s1 - s2", s1.screen_buffer - s2.screen_buffer)
                cv2.waitKey(int(10000))

            for b in buffers:
                if not np.array_equal(getattr(s1, b), getattr(s2, b)):
                    print("States after action are not equal")
                    cv2.imshow("s1", getattr(s1, b))
                    cv2.imshow("s2", getattr(s2, b))
                    cv2.imshow("s1 - s2", getattr(s1, b) - getattr(s2, b))
                    cv2.waitKey(int(10000))

                # assert np.array_equal(getattr(s1, b), getattr(s2, b))


if __name__ == "__main__":
    test_seed()
