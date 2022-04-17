import os
import numpy as np
from random import choice
import vizdoom as vzd

# This test can be run as Python script or via PyTest

def check_label(labels_buffer, label):
    # Returns True if label seems to be ok
    if label.width > 0 and label.height > 0:
        # Values decrease with the distance from the player,
        # check for >= since objects with higher values (closer ones) can obscure objects with lower values (further ones).
        detect_color = (labels_buffer >= label.value)
        detect_color[label.y:label.y + label.height, label.x:label.x + label.width] = False  # Set values inside bounding box to False
        value_outside_box = np.any(detect_color >= label.value)  # Check if there is True outside the bounding box
        value_in_box = np.any(labels_buffer[label.y:label.y + label.height, label.x:label.x + label.width] >= label.value)  # Check if there is True in the bounding box
        assert not value_outside_box and value_in_box


def test_labels_buffer():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_render_hud(False)

    game.init()

    actions = [[True, False, False, False], [False, True, False, False],
               [False, False, True, False], [False, False, False, True]]

    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        labels_buffer = state.labels_buffer
        game.make_action(choice(actions))

        for l in state.labels:
            check_label(labels_buffer, l)
    game.close()


if __name__ == '__main__':
    test_labels_buffer()