#!/usr/bin/env python3

# Test correctness of labels buffer.
# This test can be run as Python script or via PyTest

import os
from random import choice

import numpy as np

import vizdoom as vzd


def check_label(labels_buffer, label):
    assert label.value > 1
    if (
        label.width > 4 and label.height > 4
    ):  # Sometimes very tiny objects may be obscured by level geometry or not rendered due to sprite size
        # Values decrease with the distance from the player,
        # check for >= since objects with higher values (closer ones) can obscure objects with lower values (further ones).
        detect_color = labels_buffer >= label.value
        detect_color[
            label.y : label.y + label.height, label.x : label.x + label.width
        ] = False  # Set values inside bounding box to False
        value_outside_box = np.any(
            detect_color >= label.value
        )  # Check if there is True outside the bounding box
        value_in_box = np.any(
            labels_buffer[
                label.y : label.y + label.height, label.x : label.x + label.width
            ]
            >= label.value
        )  # Check if there is True in the bounding box
        assert not value_outside_box and value_in_box


def test_labels_buffer():
    print("Testing labels buffer ...")
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_render_hud(False)
    game.set_window_visible(False)
    # game.set_mode(vzd.Mode.SPECTATOR)  # For manual testing

    game.init()

    actions = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]

    game.new_episode()
    state_count = 0
    seen_labels = 0
    seen_unique_objects = set()

    while not game.is_episode_finished():
        state = game.get_state()
        labels_buffer = state.labels_buffer
        game.make_action(choice(actions))

        state_count += 1
        seen_labels += len(state.labels)
        for label in state.labels:
            seen_unique_objects.add(label.object_name)
            check_label(labels_buffer, label)
    game.close()

    print(
        f"Seen {seen_labels} labels with {len(seen_unique_objects)} unique objects in {state_count} states."
    )


if __name__ == "__main__":
    test_labels_buffer()
