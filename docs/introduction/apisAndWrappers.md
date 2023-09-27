# APIs and wrappers

ViZDoom consists of a few APIs: C++ API, Python API that is a wrapper around C++ API, and Gymnasium/Gym wrappers that wrap around Python API to allow the use of ViZDoom scenarios as Gymnasium/Gym environments.

Because ViZDoom was created before the first release of OpenAI Gym, it uses a bit different nomenclature in its API than Gym/Gymnasium:
- **environments = scenarios** - in the original ViZDoom API, environments are called scenarios,
- **observations = states** - in the original ViZDoom API, observations are called states,
- **steps = tics** - in the original ViZDoom API, steps are called tics. The Doom engine uses the name to refer to a single logic step. The tic is a single logic update of the game state that corresponds to 1/35 of a second (original Doom's framerate).


## C++ API

ViZDoom is implemented in C++ and can be used as a C++ library. The C++ API is one-to-one with the Python API. The only difference is the use of `camelCase` instead of `snake_case` for method names.

ViZDoom can be built as a static or dynamic library. The header files are located in the `include` directory.
See [examples/cpp](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/cpp) for examples, including CMake files for building.


## Python API

ViZDoom provides Python bindings for the library. The Python API is one-to-one with the C++ API. The only difference is the use of `snake_case` instead of `camelCase` for method and argument names.

There are many examples of how to use Python API in [examples/python](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python) directory.


## Gymnasium wrappers

Installing ViZDoom with `pip install vizdoom` will include
Gymnasium wrappers to interact with ViZDoom over [Gymnasium API](https://gymnasium.farama.org/).

These wrappers are under `gymnasium_wrappers`, containing the basic environment and
a few example single-player environments based on the built-in scenarios. This environment
simply initializes ViZDoom with the settings from the scenario config files
and implements the necessary API to function as a Gymnasium API.

See the following examples for use:
- [examples/python/gymnasium_wrapper.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/gymnasium_wrapper.py) for basic usage
- [examples/python/learning_stable_baselines.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/learning_stable_baselines.py) for example training with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3/) (Update - Currently facing issues, to be fixed)


## OpenAI Gym wrappers

> Gym is deprecated in favor of Gymnasium and these wrappers will be removed in the future.

Installing ViZDoom with `pip install vizdoom[gym]` will include
Gym wrappers to interact with ViZDoom over Gym API.

These wrappers are under `gym_wrappers`, containing the basic environment and
a few example environments based on the built-in scenarios. This environment
simply initializes ViZDoom with the settings from the scenario config files
and implements the necessary API to function as a Gym API.

See the following examples for use:
- [examples/python/gym_wrapper.py](https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/gym_wrapper.py) for basic usage


## Julia, Lua, and Java APIs

> Julia, Lua, and Java bindings are no longer maintained.

Julia, Lua, and Java can be found in [julia](https://github.com/Farama-Foundation/ViZDoom/tree/julia) and [java&lua](https://github.com/Farama-Foundation/ViZDoom/tree/java%26lua) branches for manual building.
