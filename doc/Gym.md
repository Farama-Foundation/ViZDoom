# OpenAI Gym wrappers

Installing ViZDoom with `pip install vizdoom[gym]` will include
Gym wrappers to interact with ViZDoom over Gym API. Note that Gym is deprecated in favour of Gymnasium and these wrappers will be removed in the future.

These wrappers are under `gym_wrappers`, containing the basic environment and
few example environments based on the built-in scenarios. This environment
simply initializes ViZDoom with the settings from the scenario config files
and implements the necessary API to function as a Gym API.

See following examples for use:
  - `examples/python/gym_wrapper.py` for basic usage
