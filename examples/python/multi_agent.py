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
from random import choice

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
    config_file = os.path.join(vzd.scenarios_path, "health_gathering.cfg")
    num_agents = 2
    episodes = 3
    max_steps_per_episode = 1000

    print(f"Starting multi-agent health gathering with {num_agents} agents")
    print(f"Running {episodes} episodes, {max_steps_per_episode} steps per episode")
    print("=" * 60)

    # Create the multi-agent environment
    try:
        env = VizdoomParallelEnv(
            config_file=config_file,
            num_agents=num_agents,
            resolution="800x600",
            timeout=max_steps_per_episode,  # This will limit steps per episode
            render_mode="human",
            use_multi_binary_action_space=False,  # Use MultiDiscrete instead of MultiBinary
            seed=42
        )
        print("Environment created successfully!")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        cleanup_environment()
        return

    # Define possible actions for each agent
    # The health gathering scenario has 3 buttons: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD
    # Since we're using MultiDiscrete action space, each action is a list of 3 discrete values (0 or 1)
    possible_actions = [
        [1, 0, 0],  # TURN_LEFT
        [0, 1, 0],  # TURN_RIGHT
        [0, 0, 1],  # MOVE_FORWARD
        [0, 0, 0],  # No action
        [1, 0, 1],  # TURN_LEFT + MOVE_FORWARD
        [0, 1, 1],  # TURN_RIGHT + MOVE_FORWARD
    ]

    try:
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1} ---")

            # Reset environment
            observations, infos = env.reset()

            # Initialize episode statistics
            episode_rewards = {agent: 0.0 for agent in env.agents}
            episode_steps = 0

            # Run episode
            while episode_steps < max_steps_per_episode:
                # Choose random actions for each agent
                actions = {}
                for agent in env.agents:
                    actions[agent] = choice(possible_actions)

                # Take step
                observations, rewards, terminations, truncations, infos = env.step(actions)

                # Update statistics
                for agent in env.agents:
                    episode_rewards[agent] += rewards[agent]

                episode_steps += 1

                # Render the environment
                try:
                    env.render()
                except Exception as e:
                    print(f"Rendering error: {e}")

                # Check if any agent is terminated or truncated
                if any(terminations.values()) or any(truncations.values()):
                    break

                # Print step info occasionally
                if episode_steps % 20 == 0:
                    print(f"  Step {episode_steps}: ", end="")
                    for agent in env.agents:
                        health = infos[agent].get('game_variables', {}).get('HEALTH', 'N/A')
                        print(f"{agent} health={health}, ", end="")
                    print()

            # Print episode results
            print(f"\nEpisode {episode + 1} Results:")
            print("-" * 40)
            for agent in env.agents:
                total_reward = episode_rewards[agent]
                health = infos[agent].get('game_variables', {}).get('HEALTH', 'N/A')
                terminated = terminations[agent]
                truncated = truncations[agent]

                print(f"  {agent}:")
                print(f"    Total Reward: {total_reward:.2f}")
                print(f"    Final Health: {health}")
                print(f"    Terminated: {terminated}")
                print(f"    Truncated: {truncated}")
                print(f"    Steps Completed: {episode_steps}")

            print("=" * 40)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup_environment()
        print("\nMulti-agent health gathering demo completed!")


if __name__ == "__main__":
    main()
