# Third-party environments

Here, we feature a selection of third-party libraries that build upon or complement ViZDoom,
offering diverse environments and tools for reinforcement learning research and development.

*Please note that the page contains environments that are not maintained by the ViZDoom Team or Farama Foundation.*

*If you have a library that you would like to see featured here, please open a pull request or an issue on the [GitHub repository](https://github.com/Farama-Foundation/ViZDoom)*


## LevDoom

[LevDoom](https://github.com/TTomilin/LevDoom) is a benchmark for generalization in pixel-based deep reinforcement learning, offering environments with difficulty levels based on visual and gameplay modifications. It consists of 4 scenarios, each with 5 difficulty levels, that modify different aspects of the base environments, such as textures, obstacles, types, sizes, and rendering of different in-game entities, etc.

LevDoom provides environments using Gymnasium API and is available through PyPi. For more details, please refer to the [CoG2022 paper](https://ieee-cog.org/2022/assets/papers/paper_30.pdf) and [GitHub repository](https://github.com/TTomilin/LevDoom).


## COOM

[COOM](https://github.com/hyintell/COOM) is a Continual Learning benchmark for embodied pixel-based RL, consisting of task sequences in visually distinct 3D environments with diverse objectives and egocentric perception. COOM is designed for task-incremental learning, in which task boundaries are clearly defined. It contains 8 scenarios, every with at least 2 difficulty levels that are combined into sequences of tasks for Continual Learning. The sequence length varies between 4, 8, and 16. COOM provides two types of task sequences:
- Cross-domain sequences compose modified versions of the same scenario (e.g., changing textures, enemy types, view height, and adding obstacles) while keeping the objective consistent.
- Cross-objective sequences contrast with Cross-Domain by changing both the visual characteristics and the objective for each task, requiring a more general policy from the agent for adequate performance.

COOM provides environments using Gymnasium API and is available through PyPi. For more details, please refer to the [GitHub repository](https://github.com/hyintell/COOM).
