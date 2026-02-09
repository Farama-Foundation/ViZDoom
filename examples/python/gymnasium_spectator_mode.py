#!/usr/bin/env python3

#####################################################################
# Example how to test Gymnasium environment yourself using ViZDoom in SPECTATOR mode.
# In SPECTATOR mode YOU play Doom and the script prints rewards and info.
#####################################################################

import gymnasium

import vizdoom as vzd
from vizdoom import gymnasium_wrapper  # noqa


if __name__ == "__main__":
    # Create the Gymnasium environment
    env = gymnasium.make(
        # Env ID, render_mode and frame_skip can be changed as needed
        "VizdoomFreedoom1E1M1-S3-v0",
        mode=vzd.Mode.SPECTATOR,  # Set spectator mode to allow human play
        window_visible=True,  # Ensure the game window is visible for spectator mode to capture inputs
        screen_resolution=vzd.ScreenResolution.RES_800X600,  # Increase screen resolution for better experience
    )

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        i = 0
        while not done:
            obs, rew, terminated, truncated, info = env.step(
                env.action_space.sample()
            )  # In spectator mode actions will be ignored, but we need to call step to advance the game (this mode is kind of hack for envs testing)
            done = terminated or truncated
            print(
                f"State #{i} | Terminated: {terminated} | Truncated: {truncated} | Reward: {rew} | Info: {info}"
            )
            i += 1

    env.close()
