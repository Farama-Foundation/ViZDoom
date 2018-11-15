#!/usr/bin/env python3

#####################################################################
# This script test basic movements
#####################################################################


from __future__ import print_function
import vizdoom as vzd
from random import random
from time import sleep
import os

GV = vzd.GameVariable
B = vzd.Button


def setup_test(buttons, variables, visible=True):
    game = vzd.DoomGame()
    game.set_doom_scenario_path("../../scenarios/basic.wad")
    game.set_doom_map("map01")
    game.set_episode_start_time(10)

    game.set_window_visible(visible)
    if visible:
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Test if everything is alright with variables
    game.set_available_game_variables(variables)
    assert len(variables) == game.get_available_game_variables_size()
    assert variables == game.get_available_game_variables()

    # Test if everything is alright with buttons
    game.set_available_buttons(buttons)
    assert len(buttons) == game.get_available_buttons_size()
    assert buttons == game.get_available_buttons()

    game.set_mode(vzd.Mode.PLAYER)

    return game


def normalize_action(action):
    for i, a in enumerate(action):
        action[i] = round(float(a), 1)
    return action


def normalize_variables(vars):
    vars = list(vars)
    for i, v in enumerate(vars):
        vars[i] = round(float(v), 1)
    return vars


def recording_test(buttons, variables, actions,
                   recording_file="test_recording.lmp", verbose=False, verbose_sleep=0.1, title="Unnamed test"):
    print("Test: {}".format(title))

    game = setup_test(buttons, variables, visible=verbose)

    # Append all zeros action
    actions.append([0.0] * game.get_available_buttons_size())

    history_of_actions = []
    history_of_variables = []

    if verbose:
        print("  Recording...")

    game.init()
    game.new_episode(recording_file)
    for action in actions:

        normalize_action(action)
        history_of_actions.append(action)
        assert type(action) == list

        state = game.get_state()
        variables = normalize_variables(state.game_variables)
        history_of_variables.append(variables)
        assert type(variables) == list

        game.make_action(action)
        last_action = game.get_last_action()
        normalize_action(last_action)
        assert type(last_action) == list

        if verbose:
            print("  State: {}, tic: {}, action: {}, last action: {}, variables: {}".format(state.number,
                    state.tic, action, last_action, variables))
            sleep(verbose_sleep)

        # Asserts
        assert action == last_action

    game.close()

    if verbose:
        print("  Replaying...")

    state_number = 0
    game.init()
    game.replay_episode(recording_file)
    while not game.is_episode_finished() and state_number < len(history_of_variables):
        state = game.get_state()
        variables = normalize_variables(state.game_variables)
        assert type(variables) == list

        game.advance_action()
        last_action = game.get_last_action()
        normalize_action(last_action)
        assert type(last_action) == list

        sleep(0.0001)

        if verbose:
            print("  State: {}, tic: {}, history action: {}, last action: {}, history variables: {} variables: {}".format(state.number,
                state.tic, history_of_actions[state_number], last_action, history_of_variables[state_number], variables))
            sleep(verbose_sleep)

        # Asserts
        try:
            assert history_of_variables[state_number] == variables
            assert history_of_actions[state_number] == last_action
        except AssertionError:
            print("  Test failed:")
            print("    State: {}, tic: {}, history action: {} != last action: {} or history variables: {} != variables: {}".format(state.number,
                state.tic, history_of_actions[state_number], last_action, history_of_variables[state_number], variables))

        state_number += 1

    game.close()
    os.remove(recording_file)


def random_action(action_len, button_press_prob):
    new_action = [0] * action_len
    for i in range(action_len):
        new_action[i] = 1.0 if random() < button_press_prob else 0.0
    return new_action


def random_test():
    buttons = [B.MOVE_LEFT,
               B.MOVE_RIGHT,
               B.MOVE_BACKWARD,
               B.MOVE_FORWARD,
               B.TURN_RIGHT,
               B.TURN_LEFT,
               B.CROUCH,
               B.JUMP]

    variables = [GV.POSITION_X,
                 GV.POSITION_Y,
                 GV.POSITION_Z,
                 GV.VIEW_HEIGHT,
                 GV.ANGLE,
                 GV.PITCH,
                 GV.ROLL]

    test_actions1 = []
    for _ in range(1000):
        test_actions1.append(random_action(len(buttons), 0.5))

    test_actions2 = []
    for _ in range(1000):
        test_actions2.append(random_action(len(buttons), 0.9))

    test_actions3 = []
    for _ in range(1000):
        test_actions3.append(random_action(len(buttons), 0.1))


    recording_test(buttons, variables, test_actions1, title="Random test #1")
    recording_test(buttons, variables, test_actions2, title="Random test #2")
    recording_test(buttons, variables, test_actions3, title="Random test #3")


# https://github.com/mwydmuch/ViZDoom/issues/354
def ankitkv_test():
    buttons = [B.MOVE_FORWARD, B.TURN_LEFT_RIGHT_DELTA]
    variables = [GV.POSITION_X, GV.ANGLE]

    test_actions1 = [
        [0.0, 90.0],
        [0.0, -90.0],
        [1.0, 0.0],
        [0.0, 90.0],
        [0.0, -90.0],
        [0.0, 90.0],
        [0.0, -90.0],
        [0.0, 90.0],
        [0.0, 90.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, -90.0],
        [0.0, 90.0],
        [1.0, 0.0],
    ]

    test_actions2 = [
        [0.0, -90.0],
        [0.0, 90.0],
        [1.0, 0.0],
        [0.0, -90.0],
        [0.0, 90.0],
        [0.0, -90.0],
        [0.0, 90.0],
        [0.0, -90.0],
        [0.0, -90.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 90.0],
        [0.0, -90.0],
        [1.0, 0.0],
    ]

    recording_test(buttons, variables, test_actions1, title="ankitkv test #1")
    recording_test(buttons, variables, test_actions2, title="ankitkv test #2")


def delta_buttons_test():
    buttons = [B.MOVE_FORWARD_BACKWARD_DELTA,
               B.MOVE_LEFT_RIGHT_DELTA,
               B.LOOK_UP_DOWN_DELTA,
               B.TURN_LEFT_RIGHT_DELTA]

    variables = [GV.POSITION_X,
                 GV.POSITION_Y,
                 GV.ANGLE,
                 GV.PITCH]

    test_actions1 = []
    for i in range(10):
        new_action = [25, 15 if i % 2 == 0 else 0, -3, 10]
        test_actions1.append(new_action)

    for i in range(10):
        new_action = [-15, -10 if i % 2 == 0 else 0, 6, -20]
        test_actions1.append(new_action)

    recording_test(buttons, variables, test_actions1, title="Delta buttons test #1")


if __name__ == "__main__":
    random_test()
    ankitkv_test()
    delta_buttons_test()
