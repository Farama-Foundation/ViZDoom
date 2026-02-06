#!/usr/bin/env python3

######################################################
# Script to create GIFs for ViZDoom Gymnasium environments
######################################################

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np

import vizdoom.gymnasium_wrapper  # noqa


def save_gif(frames, gif_path, fps=35):
    """Save a list of frames as a GIF."""
    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"GIF saved to {gif_path}")
    else:
        print("No frames to save for GIF.")


def create_env_gif(env_name, duration=5, fps=35, seed=1993):
    """Create a GIF for a single ViZDoom environment."""
    try:
        print(f"Creating GIF for {env_name}...")
        env_doom_graphics = gym.make(env_name)
        env_doom_graphics.reset(seed=seed)
        env_freedoom_graphics = gym.make(env_name)
        env_freedoom_graphics.reset(seed=seed)

        frames_doom_graphics = []
        frames_freedoom_graphics = []
        total_frames = duration * fps

        for _ in range(total_frames):
            action = env_doom_graphics.action_space.sample()
            obs_doom_graphics, _, terminated, truncated, _ = env_doom_graphics.step(action)
            obs_freedoom_graphics, _, terminated, truncated, _ = env_freedoom_graphics.step(action)

            # Get RGB frame
            frames_doom_graphics.append(obs_doom_graphics['screen'])
            frames_freedoom_graphics.append(obs_freedoom_graphics['screen'])

            if terminated or truncated:
                env_doom_graphics.reset(seed=seed)
                env_freedoom_graphics.reset(seed=seed)

        env_doom_graphics.close()
        env_freedoom_graphics.close()

        if frames_doom_graphics:
            gif_path = Path(__file__).parent / "../docs/_static/img" / f"{env_name.lower().split("-")[0].replace('vizdoom', 'vizdoom-')}-doom.gif"
            save_gif(frames_doom_graphics, gif_path, fps=fps)
        if frames_freedoom_graphics:
            gif_path = Path(__file__).parent / "../docs/_static/img" / f"{env_name.lower().split('-')[0].replace('vizdoom', 'vizdoom-')}-freedoom.gif"
            save_gif(frames_freedoom_graphics, gif_path, fps=fps)

    except Exception as e:
        print(f"Failed to create GIF for {env_name}: {e}")


def main():
    # List of ViZDoom Gymnasium environments for which to create GIFs
    vizdoom_envs = [
        "VizdoomBasic-v1",
        "VizdoomBasicAudio-v1",
        "VizdoomBasicNotifications-v1",
        "VizdoomCorridor-v1",
        "VizdoomDefendCenter-v1",
        "VizdoomDefendLine-v1",
        "VizdoomHealthGathering-v1",
        "VizdoomMyWayHome-v1",
        "VizdoomPredictPosition-v1",
        "VizdoomTakeCover-v1",
        "VizdoomDeathmatch-v1",
        "VizdoomHealthGatheringSupreme-v1"
    ]

    # Set random seed
    np.random.seed(1993)

    # Create GIFs for all environments
    for env_name in vizdoom_envs:
        create_env_gif(env_name, duration=10, seed=1993)


if __name__ == "__main__":
    main()
