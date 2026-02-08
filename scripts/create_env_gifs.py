#!/usr/bin/env python3

######################################################
# Script to create GIFs for ViZDoom Gymnasium environments
######################################################

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np

import vizdoom.gymnasium_wrapper  # noqa


def save_gif(frames, gif_path: Path, fps: float = 20):
    """Save a list of frames as a GIF."""
    if len(frames) > 0:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"GIF saved to {gif_path}")
    else:
        print("No frames to save for GIF.")


def create_env_gif(
    env_name,
    duration: int = 3,
    fps: int = 20,
    seed: int = 1993,
    drop: int = 4,
):
    """Create a GIF for a single ViZDoom environment."""
    try:
        print(f"Creating GIF for {env_name}...")
        if drop <= 0:
            raise ValueError("drop must be greater than 0")

        env_doom_graphics = gym.make(
            env_name,
            doom_game_path="/home/marek/Workspace/ViZDoom_dev_mode/bin/python3.13/vizdoom/doom2.wad",
        )
        env_doom_graphics.reset(seed=seed)
        env_freedoom_graphics = gym.make(
            env_name,
            doom_game_path="/home/marek/Workspace/ViZDoom_dev_mode/bin/python3.13/vizdoom/freedoom2.wad",
        )
        env_freedoom_graphics.reset(seed=seed)

        frames_doom_graphics = []
        frames_freedoom_graphics = []
        total_frames = duration * fps

        for _ in range(total_frames):
            action = env_doom_graphics.action_space.sample()
            obs_doom_graphics, _, terminated, truncated, _ = env_doom_graphics.step(
                action
            )
            (
                obs_freedoom_graphics,
                _,
                terminated,
                truncated,
                _,
            ) = env_freedoom_graphics.step(action)

            # Get RGB frame
            frames_doom_graphics.append(obs_doom_graphics["screen"])
            frames_freedoom_graphics.append(obs_freedoom_graphics["screen"])

            if terminated or truncated:
                env_doom_graphics.reset(seed=seed)
                env_freedoom_graphics.reset(seed=seed)

        env_doom_graphics.close()
        env_freedoom_graphics.close()
        frames_doom_graphics = frames_doom_graphics[::drop]
        frames_freedoom_graphics = frames_freedoom_graphics[::drop]
        output_fps = fps / drop
        env_prefix = env_name.split("-", 1)[0]

        if len(frames_doom_graphics) > 0:
            gif_path = (
                Path(__file__).parent
                / "../docs/_static/img/envs"
                / f"{env_prefix}-Doom2.gif"
            )
            save_gif(frames_doom_graphics, gif_path, fps=output_fps)
        if len(frames_freedoom_graphics) > 0:
            gif_path = (
                Path(__file__).parent
                / "../docs/_static/img/envs"
                / f"{env_prefix}-Freedoom2.gif"
            )
            save_gif(frames_freedoom_graphics, gif_path, fps=output_fps)

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
        "VizdoomHealthGatheringSupreme-v1",
        # TODO?
        # "VizdoomDoomE1M1-S3-v0",
        # "VizdoomDoom2MAP01-S3-v0",
        # "VizdoomFreedoom1E1M1-S3-v0",
        # "VizdoomFreedoom2MAP01-S3-v0",
    ]

    parser = argparse.ArgumentParser(description="Create docs GIFs for ViZDoom envs.")
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Seconds per GIF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="GIF frame rate.",
    )
    parser.add_argument(
        "--drop",
        "-d",
        type=int,
        default=4,
        help="Drop every n frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993 + 42,
        help="Random seed for action sampling.",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create GIFs for all environments
    for env_name in vizdoom_envs:
        create_env_gif(
            env_name,
            duration=args.duration,
            fps=args.fps,
            seed=args.seed,
            drop=args.drop,
        )


if __name__ == "__main__":
    main()
