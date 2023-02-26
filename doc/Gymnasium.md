# Gymnasium wrappers

Installing ViZDoom with `pip install vizdoom[gymnasium]` will include
Gymnasium wrappers to interact with ViZDoom over [Gymnasium API](https://gymnasium.farama.org/).

These wrappers are under `gymnasium_wrappers`, containing the basic environment and
few example environments based on the built-in scenarios. This environment
simply initializes ViZDoom with the settings from the scenario config files
and implements the necessary API to function as a Gymnasium API.

See following examples for use:
  - `examples/python/gymnasium_wrapper.py` for basic usage
  - `examples/python/learning_stable_baselines.py` for example training with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3/) (Update - Currently facing issues, to be fixed)
