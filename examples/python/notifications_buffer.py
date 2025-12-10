#!/usr/bin/env python3

#####################################################################
# This script presents how to access notification buffer.
#
# Notifications buffer was added in version 1.3.0
#####################################################################

import os
from random import choice

import vizdoom as vzd


if __name__ == "__main__":
    game = vzd.DoomGame()

    # Load config of the basic scenario
    game.load_config(os.path.join(vzd.scenarios_path, "basic_notifications.cfg"))

    # Turns on the notifications buffer. (turned off by default)
    game.set_notifications_buffer_enabled(True)

    frameskip = 4
    game.set_notifications_buffer_size(
        frameskip
    )  # From how many ticks back notifications are stored

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    episodes = 3
    game.init()
    for i in range(episodes):
        print(f"Episode #{i + 1}")

        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()
            assert state is not None

            # Print notifications buffer
            notifications_buffer = state.notifications_buffer
            if notifications_buffer and len(notifications_buffer) > 0:
                print(f"Notifications buffer: {notifications_buffer}")

            # Makes a random action and get remember reward.
            r = game.make_action(choice(actions), frameskip)

    game.close()
