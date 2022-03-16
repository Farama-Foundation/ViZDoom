# OpenAI Gym wrappers

Installing ViZDoom with `pip install vizdoom[gym]` will include
Gym wrappers to interact with ViZDoom over [Gym API](https://www.gymlibrary.ml/).

These wrappers are under `gym_wrappers`, containing the basic environment and
few example environments based on the built-in scenarios. This environment
simply initializes ViZDoom with the settings from the scenario config files
and implements the necessary API to function as a Gym API.

See following examples for use:
  - `examples/python/gym_wrapper.py` for basic usage
  - `examples/python/learning_stable_baselines.py` for example training with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3/)
