# Tests for category mapping functionality (semantic segmentation).

import glob
import os
import random

import vizdoom


def custom_categories_test(scenario: str = "scenarios/defend_the_line.wad"):
    print(f"Testing custom category mapping on {scenario}...")

    # Create game1 instance
    game = vizdoom.DoomGame()

    # Set up basic configuration
    game.set_doom_scenario_path(scenario)
    game.set_window_visible(False)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    game.set_labels_buffer_enabled(True)
    game.init()

    # Start a new episode
    random.seed("ViZDoom!")
    game.set_seed(random.randrange(0, 256))
    game.new_episode()

    doom_categories = vizdoom.get_default_categories()
    seen_objects = set()
    seen_categories = set()

    # Move randomly for a while to generate some labels
    action_size = [0] * game.get_available_buttons_size()
    for _ in range(2000):
        game.make_action([random.random() > 0.5 for _ in action_size], 4)

        # Get the state and check labels
        state = game.get_state()
        if state and state.labels:
            labels = sorted(state.labels, key=lambda label: label.object_name)
            for label in labels:
                if label.object_category == "Self":
                    assert (
                        label.object_name == "DoomPlayer"
                    ), f'Assigned "Self" to non-DoomPlayer object: {label.object_name}'
                else:
                    assert (
                        label.object_category in doom_categories
                    ), f'Unknown category "{label.object_category}" assigned to {label.object_name}'
                seen_objects.add(label.object_name)
                seen_categories.add(label.object_category)
        else:
            game.new_episode()

    # Close the game
    game.close()
    print(
        f"Seen objects: {', '.join(sorted(seen_objects))}\nSeen categories: {', '.join(sorted(seen_categories))}\n"
    )
    assert "Unknown" not in seen_categories, '"Unknown" category in default scenarios!'


def test_object_categories():
    scenario_pattern = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scenarios", "*.wad"
    )
    print(f"Finding scenarios in {scenario_pattern}")
    for scenario in sorted(glob.glob(scenario_pattern)):
        if any(multi_keyword in scenario for multi_keyword in ["multi", "cig"]):
            continue
        custom_categories_test(scenario)
    print("\nTest completed!")
