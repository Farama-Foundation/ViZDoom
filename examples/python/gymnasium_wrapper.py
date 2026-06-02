#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a Gymnasium env
#####################################################################

import gymnasium


from vizdoom import gymnasium_wrapper  # noqa



if __name__ == "__main__":
    env = gymnasium.make(
        "VizdoomHealthGatheringSupreme-v1", render_mode="human", frame_skip=4
    )

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs, info = env.reset()
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            env.render()
            done = terminated or truncated
