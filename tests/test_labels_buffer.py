#!/usr/bin/env python3

# Test correctness of labels buffer.
# This test can be run as Python script or via PyTest

import os
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest

import vizdoom as vzd


def _check_label(labels_buffer, label):
    assert label.value > 1
    if (
        label.width > 4 and label.height > 4
    ):  # Sometimes very tiny objects may be obscured by level geometry or not rendered due to sprite size
        box = np.s_[label.y : label.y + label.height, label.x : label.x + label.width]
        # Only this object's pixels must stay inside its bounding box.
        outside = labels_buffer == label.value
        outside[box] = False
        value_outside_box = np.any(outside)
        # Closer objects can obscure this object's pixels inside the box.
        value_in_box = np.any(labels_buffer[box] >= label.value)
        if value_outside_box or not value_in_box:
            raise AssertionError(
                f"object={label.object_name}, value={label.value}, "
                f"box=({label.x}, {label.y}, {label.width}, {label.height}), "
                f"inside_values={np.unique(labels_buffer[box]).tolist()}, "
                f"outside_pixels={np.argwhere(outside).tolist()}"
            )


@pytest.mark.parametrize("pixel", [(2, 7), (7, 2)])
def test_label_pixels_outside_box(pixel):
    label = SimpleNamespace(
        value=10, x=2, y=2, width=5, height=5, object_name="test_object"
    )
    buffer = np.zeros((10, 10), dtype=np.uint8)
    buffer[3, 3] = label.value
    buffer[pixel] = label.value
    with pytest.raises(AssertionError, match="outside_pixels"):
        _check_label(buffer, label)


def test_label_box_edges_and_other_objects():
    label = SimpleNamespace(
        value=10, x=2, y=2, width=6, height=6, object_name="test_object"
    )
    buffer = np.zeros((10, 10), dtype=np.uint8)
    buffer[2, 7] = label.value
    buffer[7, 2] = label.value
    buffer[0, 0] = label.value + 1
    _check_label(buffer, label)


# Seed 63 exposes a ShellBox at the screen edge whose only remaining labeled
# pixels are in the last row/column of its bounding box (state 360).
@pytest.mark.parametrize("seed", [0, 1, 2, 63])
def test_labels_buffer(seed):
    print("Testing labels buffer ...")
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_render_hud(False)
    game.set_window_visible(False)
    # game.set_mode(vzd.Mode.SPECTATOR)  # For manual testing

    game.set_seed(seed)
    rng = Random(seed)
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

    try:
        while not game.is_episode_finished():
            state = game.get_state()
            assert state is not None
            labels_buffer = state.labels_buffer

            state_count += 1
            seen_labels += len(state.labels)
            for label in state.labels:
                seen_unique_objects.add(label.object_name)
                try:
                    _check_label(labels_buffer, label)
                except AssertionError as error:
                    raise AssertionError(
                        f"seed={seed}, state={state_count}: {error}"
                    ) from error
            game.make_action(rng.choice(actions))
    finally:
        game.close()

    print(
        f"Seen {seen_labels} labels with {len(seen_unique_objects)} unique objects in {state_count} states."
    )


if __name__ == "__main__":
    for seed in [0, 1, 2, 63]:
        test_labels_buffer(seed)
