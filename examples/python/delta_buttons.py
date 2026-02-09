#!/usr/bin/env python3

#####################################################################
# This script presents how to use delta buttons (continuous buttons).
# Delta buttons are used to control the precision/speed of an action.
# For example, they can be used to control the speed of movement or the angle of turning.
#####################################################################

from time import sleep

import vizdoom as vzd


if __name__ == "__main__":

    game = vzd.DoomGame()

    # Adds delta buttons that will be allowed and set the maximum allowed value (optional).
    game.add_available_button(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 10)
    game.add_available_button(vzd.Button.MOVE_LEFT_RIGHT_DELTA, 5)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 5)
    game.add_available_button(vzd.Button.LOOK_UP_DOWN_DELTA)

    # For normal buttons (binary) all values other than 0 are interpreted as pushed.
    # For delta buttons values determine a precision/speed.
    #
    # For TURN_LEFT_RIGHT_DELTA and LOOK_UP_DOWN_DELTA value is the angle (in degrees)
    # of which the viewing angle will change.
    #
    # For MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA (rarely used)
    # value is the speed of movement in a given direction (100 is close to the maximum speed).
    action = [100, 10, 10, 1]  # floating point values can be used

    # If button's absolute value > max button's value then value = max value with original value sign.

    # Delta buttons in spectator modes correspond to mouse movements.
    # Maximum allowed values also apply to spectator modes.
    # game.add_game_args("+freelook 1")    # Use this to enable looking around with the mouse.
    # game.set_mode(Mode.SPECTATOR)

    game.set_window_visible(True)

    game.init()

    episodes = 10
    sleep_time = 0.028

    for i in range(episodes):
        print(f"Episode #{i + 1}")

        game.new_episode()

        while not game.is_episode_finished():

            state = game.get_state()
            assert state is not None
            reward = game.make_action(action)

            time = game.get_episode_time()

            action[0] = time % 100 - 50
            action[1] = time % 100 - 50
            action[2] = time % 100 - 50
            if not time % 50:
                action[3] = -action[3]

            print(f"State #{state.number} | Action made: {action}")

            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished.")
        print("************************")
