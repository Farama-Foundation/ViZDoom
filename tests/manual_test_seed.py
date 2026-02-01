#!/usr/bin/env python3ch

# Tests ViZDoom seed option.
# This test can be run as Python script or via PyTest

import copy
import itertools
import os
import random

import cv2
import numpy as np

import vizdoom as vzd


def test_seed(
    repeats=200,
    tics=34,
    audio_buffer=False,
    test_only_animated_textures=False,
    seed=1993,
):
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
    scenarios_with_animated_textures = [
        "health_gathering.cfg",
        "health_gathering_supreme.cfg",
        "deathmatch.cfg",
    ]
    configs = [
        file
        for file in os.listdir(vzd.scenarios_path)
        if file.endswith(".cfg")
        and file not in scenarios_to_skip
        and (
            not test_only_animated_textures or file in scenarios_with_animated_textures
        )
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
        game.set_episode_start_time(14)

        # Creates all possible actions depending on how many buttons there are.
        actions_num = game.get_available_buttons_size()
        actions = []
        for perm in itertools.product([False, True], repeat=actions_num):
            actions.append(list(perm))

        # Enable all buffers
        buffers = ["screen_buffer", "depth_buffer", "labels_buffer", "automap_buffer"]
        # buffers = ["screen_buffer"]
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

            initial_states.append(copy.deepcopy(game.get_state()))
            # This sometimes fails with animated textures (some dependency on number of updates?)
            # if i % 2 == 0:
            #     game.make_action(random.choice(actions), tics=tics)
            # else:
            #     action = random.choice(actions)
            #     for _ in range(tics):
            #         game.make_action(action, tics=1)

            game.make_action(random.choice(actions), tics=tics)
            states_after_action.append(copy.deepcopy(game.get_state()))

        game.close()

        failed = False
        for i, (s1, s2) in enumerate(zip(initial_states[:-1], initial_states[1:])):
            assert s1.tic == s2.tic
            assert np.array_equal(s1.game_variables, s2.game_variables)

            if not np.array_equal(s1.screen_buffer, s2.screen_buffer):
                print(f"Initial states are not equal after repeat {i}")
                print(f"s1: {s1.tic}, {s1.game_variables}")
                print(f"s2: {s2.tic}, {s2.game_variables}")
                # print(np.all(s1.screen_buffer == s2.screen_buffer))
                # print(np.array_equal(s1.screen_buffer, s2.screen_buffer))
                cv2.imshow("s1", s1.screen_buffer)
                cv2.imshow("s2", s2.screen_buffer)
                cv2.imshow("s1 - s2", s1.screen_buffer - s2.screen_buffer)
                cv2.waitKey(int(10000))
                failed = True

            for b in buffers:
                if not np.array_equal(getattr(s1, b), getattr(s2, b)):
                    print(
                        f"Initial states are not equal after repeat {i} for buffer {b}"
                    )
                    cv2.imshow("s1", getattr(s1, b))
                    cv2.imshow("s2", getattr(s2, b))
                    cv2.imshow("s1 - s2", getattr(s1, b) - getattr(s2, b))
                    cv2.waitKey(int(10000))
                    failed = True

            if failed:
                break

        failed = False
        for i, (s1, s2) in enumerate(
            zip(states_after_action[:-1], states_after_action[1:])
        ):
            assert s1.tic == s2.tic
            assert np.array_equal(s1.game_variables, s2.game_variables)

            if not np.array_equal(s1.screen_buffer, s2.screen_buffer):
                print(f"States after action are not equal after repeat {i}")
                print(f"s1: {s1.tic}, {s1.game_variables}")
                print(f"s2: {s2.tic}, {s2.game_variables}")
                # print(np.all(s1.screen_buffer == s2.screen_buffer))
                # print(np.array_equal(s1.screen_buffer, s2.screen_buffer))
                cv2.imshow("s1", s1.screen_buffer)
                cv2.imshow("s2", s2.screen_buffer)
                cv2.imshow("s1 - s2", s1.screen_buffer - s2.screen_buffer)
                cv2.waitKey(int(10000))
                failed = True

            for b in buffers:
                if not np.array_equal(getattr(s1, b), getattr(s2, b)):
                    print(
                        f"States after action are not equal after repeat {i} for buffer {b}"
                    )
                    cv2.imshow("s1", getattr(s1, b))
                    cv2.imshow("s2", getattr(s2, b))
                    cv2.imshow("s1 - s2", getattr(s1, b) - getattr(s2, b))
                    cv2.waitKey(int(10000))
                    failed = True

            if failed:
                break


if __name__ == "__main__":
    test_seed()
