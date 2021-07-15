#!/usr/bin/env python3

#####################################################################
# This script presents how to read and use the sound buffer.
# This script stores a "basic_sounds.wav" file of recorded audio.
# Note: This requires scipy library
#####################################################################

import vizdoom as vzd

from random import choice
import numpy as np
from scipy.io import wavfile
from time import sleep

if __name__ == "__main__":
    game = vzd.DoomGame()
    game.set_doom_scenario_path("../../scenarios/basic.wad")
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    game.set_doom_map("map01")
    game.set_mode(vzd.Mode.PLAYER)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(True)
    # Turns on the sound in the observation space. (turned off by default)
    # NOTE: cant be positive when game.set_sound_enabled(False)
    # If this is switched on, the audio will stop playing on device
    SOFT_SOUND_ENABLED = False
    game.set_soft_sound_enabled(SOFT_SOUND_ENABLED)

    # Set the sampling rate used in the observation window. Has to be one from:
    # - vzd.SamplingRate.SR_44100 (default)
    # - vzd.SamplingRate.SR_22050
    # - vzd.SamplingRate.SR_11025
    game.set_sound_sampling_freq(vzd.SamplingRate.SR_22050)

    # When using frameskip (`tics` parameter of the `make_actions` function),
    # we would only get the latest segment of audio (1/35 seconds).
    # With this function you can set how many frames of audio you want to store,
    # so you can get all audio that happened during the frameskip
    frameskip = 4
    game.set_sound_observation_num_frames(frameskip)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    episodes = 3
    audio_slices = []
    for i in range(episodes):
        print("Episode #" + str(i + 1))
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            audio_buffer = state.audio_buffer
            audio_slices.append(audio_buffer.copy())

            # Makes a random action and get remember reward.
            r = game.make_action(choice(actions), frameskip)

            if not SOFT_SOUND_ENABLED:
                sleep(sleep_time * frameskip)            
    game.close()

    # Save audio file
    wavfile.write("basic_sounds.wav", 22050, np.concatenate(audio_slices, axis=0))
