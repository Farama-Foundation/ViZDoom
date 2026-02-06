#!/usr/bin/env python3

#####################################################################
# Test and benchmark the vectorization of VizDoom in gymnasium
#####################################################################

import argparse
import time
import warnings

import gymnasium

from vizdoom import gymnasium_wrapper  # noqa


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--n_envs", type=int, default=1, help="Number of envs")
parser.add_argument(
    "--mode", type=str, default="async", help="Gymnasium vectorization mode"
)
args = parser.parse_args()
seed = 42
n_steps = 1000


if __name__ == "__main__":

    # Pick an environment VizdoomCorridor-v1
    envs = gymnasium.make_vec(
        "VizdoomCorridor-v1", num_envs=args.n_envs, vectorization_mode=args.mode
    )

    # Time it
    start = time.time()

    observation, info = envs.reset()
    for _ in range(n_steps):
        # No learning here, for purposes of benchmarks
        actions = envs.action_space.sample()
        observations, rewards, terminations, truncations, infos = envs.step(actions)
        # no need for env.reset() here since the default is AutoReset(https://farama.org/Vector-Autoreset-Mode)
        # if terminated or truncated:
        #    observation, info = env.reset()
    print(f"{args.n_envs}  {n_steps * args.n_envs / round(time.time() - start, 1)}")

    envs.close()
