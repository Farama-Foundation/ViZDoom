#!/usr/bin/env python3

#####################################################################
# This script presents how to read and use the sound buffer.
# This script stores a "basic_sounds.wav" file of recorded audio.
# Note: This requires scipy library
#####################################################################

import vizdoom as vzd

import os
from random import choice
import numpy as np
from scipy.io import wavfile
from time import sleep


if __name__ == "__main__":
    game = vzd.DoomGame()

    # Load config of the basic scenario
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))

    # Turns on the audio buffer. (turned off by default)
    # If this is switched on, the audio will stop playing on device, even with game.set_sound_enabled(True)
    # Setting game.set_sound_enabled(True) is not required for audio buffer to work.
    AUDIO_BUFFER_ENABLED = True
    game.set_audio_buffer_enabled(AUDIO_BUFFER_ENABLED)

    # Set the sampling rate used in the observation window. Has to be one from:
    # - vzd.SamplingRate.SR_44100 (default)
    # - vzd.SamplingRate.SR_22050
    # - vzd.SamplingRate.SR_11025
    # Remember to also set audio saving code at the bottom to use same sampling rate!
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)

    # When using frameskip (`tics` parameter of the `make_actions` function),
    # we would only get the latest "frame" of audio (1/35 seconds).
    # With this function you can set how many last "frames" of audio will be stored in audio buffer.
    # Note that if you use larger frameskip than size of audio buffer you will lost some information about the audio.
    # If you use frameskip smaller than size of audio buffer, some audio information will overlap.
    frameskip = 4
    game.set_audio_buffer_size(frameskip)

    # This could fix "no audio in buffer" bug on Ubuntu 20.04.
    #game.add_game_args("+snd_efx 0")

    # Initialize the game. Further configuration won't take any effect from now on.
    try:
        game.init()
    except Exception as e:
        print(
            "[ERROR] Could not launch ViZDoom. If you see an error above about BiquadFilter and gain,\n"
            "        try setting game.add_game_args('+snd_efx 0'). If that fails, see\n"
            "        https://github.com/mwydmuch/ViZDoom/pull/486"
        )
        exit(1)

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
            audio_slices.append(audio_buffer)

            # Makes a random action and get remember reward.
            r = game.make_action(choice(actions), frameskip)

            if not AUDIO_BUFFER_ENABLED:
                sleep(sleep_time * frameskip)            
    game.close()

    if AUDIO_BUFFER_ENABLED:
        # Check that we have audio (having no audio is a common bug, see
        # https://github.com/mwydmuch/ViZDoom/pull/486
        audio_data = np.concatenate(audio_slices, axis=0)
        if audio_data.max() == 0:
            print(
                "[WARNING] Audio buffers were full of silence. This is a common bug on e.g. Ubuntu 20.04\n"
                "          See https://github.com/mwydmuch/ViZDoom/pull/486\n"
                "          Two possible fixes:\n"
                "            1) Try setting game.add_game_args('+snd_efx 0'). This my disable some audio effects\n"
                "            2) Try installing a newer version of OpenAL Soft library, see https://github.com/mwydmuch/ViZDoom/pull/486#issuecomment-889389185"
            )
        # Save audio file
        wavfile.write("basic_sounds.wav", 22050, np.concatenate(audio_slices, axis=0))
