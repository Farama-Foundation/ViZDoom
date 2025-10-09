#!/usr/bin/env python3

#####################################################################
# This script demonstrates how to use the multi-agent PettingZoo wrapper
# for ViZDoom. It runs the health gathering scenario with multiple agents
# for 3 episodes, 1000 steps per episode, and renders the screen.
# Results are printed after each episode for each agent.
#####################################################################

import atexit
import os
import signal
import sys
from pathlib import Path
from random import choice

# Add to Python path as pettingzoo_wrapper in root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pettingzoo_wrapper import make

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import vizdoom as vzd
from pettingzoo_wrapper.base_pettingzoo_env import VizdoomParallelEnv

# Global environment variable for cleanup
env = None


def cleanup_environment():
    """Clean up the environment and ensure all processes are terminated."""
    global env
    if env is not None:
        try:
            print("\nCleaning up environment...")
            env.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            env = None


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure proper cleanup."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup_environment()
    sys.exit(1)


def main():
    global env

    # Register signal handlers for proper cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_environment)

    # Configuration
    scenario = "pitfall"
    num_agents = 2
    episodes = 3

    print(f"Starting multi-agent {scenario} with {num_agents} agents")
    print(f"Running {episodes} episodes")
    print("=" * 60)

    # Create the multi-agent environment
    try:
        env = make(
            scenario=scenario,
            num_agents=num_agents,
            resolution="800x600",
            render_mode="human",
            seed=42,
            netmode=1,
            skip_frames=1,
            async_mode=True,
            ticrate=20,
        )
        print("Environment created successfully!")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        cleanup_environment()
        return

    try:
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1} ---")

            episode_key = f"episode_{episode + 1}"
            episode_buffer = {episode_key: []}
            last_infos = None

            # Reset environment
            env.reset()

            # Initialize episode statistics
            episode_rewards = {agent: 0.0 for agent in env.agents}
            episode_steps = 0
            done = False

            # Run episode
            while not done:
                # Choose random actions for each agent
                actions = {}
                for agent in env.agents:
                    actions[agent] = env.action_space(agent).sample()

                # Take step
                observations, rewards, terminations, truncations, infos = env.step(actions)

                last_infos = infos
                step_record = {}

                # Update statistics
                for i, agent in enumerate(env.agents):
                    episode_rewards[agent] += rewards[agent]
                    info = infos.get(agent, {}) or {}

                    step_record[agent] = {
                        "step": info.get("step", episode_steps),
                        "dead": int(info.get("DEAD", 0)),
                        "position_x": info.get("POSITION_X", None),
                        "rewards": float(rewards.get(agent, 0.0)),              # single reward for this step
                        "actions": actions.get(agent, None),                    # single action for this step
                        "observations": observations.get(agent, None),          # single obs for this step
                    }

                episode_buffer[episode_key].append(step_record)

                episode_steps += 1

                # Render the environment
                try:
                    env.render()
                except Exception as e:
                    print(f"Rendering error: {e}")

                # Check if any agent is terminated or truncated
                done = any(terminations.values()) or any(truncations.values())

            # Print episode results
            print(f"\nEpisode {episode + 1} Results:")
            print("-" * 40)
            for agent in env.agents:
                total_reward = episode_rewards[agent]
                terminated = terminations[agent]
                truncated = truncations[agent]

                print(f"  {agent}:")
                print(f"    Total Reward: {total_reward:.2f}")
                print(f"    Terminated: {terminated}")
                print(f"    Truncated: {truncated}")
                print(f"    Steps Completed: {episode_steps}")
                print(f"    Agent steps: {last_infos.get(agent, {}).get('step', 'N/A') if last_infos else 'N/A'}")

            print("=" * 40)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup_environment()
        print(f"\nMulti-agent {scenario} example completed!")


if __name__ == "__main__":
    main()
