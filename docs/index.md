---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/vizdoom-text.png
:alt: Gymnasium Logo
```

```{project-heading}
Library for developing AI bots that play Doom using visual information.
```

```{figure} _static/img/vizdoom-demo.gif
   :alt: ViZDoom Demo
```

This library allows creating of environments based on the Doom engine. It is primarily intended for research in machine visual learning and deep reinforcement learning, in particular. The library is written in C++ and provides Python API and wrappers for Gymnasium/OpenAI Gym interface. It is multi-platform (Linux, macOS, Windows), lightweight (just a few MB), and fast (capable of rendering even 7000 fps on a single CPU thread). The design of the library is meant to give high customization options; it supports single-player as well as multi-player modes and implements many additional features like access to the depth buffer (3D vision), automatic labeling of game objects visible in the frame, access to the audio buffer, access to the list of actors/objects and map geometry, off-screen rendering, episodes recording, in-game time scaling. While it provides a set of simple exemplary scenarios, it easily allows one to create custom ones thanks to the visual editors and scripting language supported by the engine. ViZDoom is based on [ZDoom](https://zdoom.org) source-port to provide the game mechanics.


The Gymnasium interface allows to initialize and interact with the ViZDoom default environments as follows:

```{code-block} python
import gymnasium
from vizdoom import gymnasium_wrapper
env = gymnasium.make("VizdoomDeadlyCorridor-v0")
observation, info = env.reset()
for _ in range(1000):
   action = policy(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
```

There is also the original ViZDoom API:

```{code-block} python
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
game.init()
for _ in range(1000):
   state = game.get_state()
   action = policy(state)  # this is where you would insert your policy
   reward = game.make_action(action)

   if game.is_episode_finished():
      game.new_episode()

game.close()
```


```{toctree}
:hidden:
:caption: Introduction

introduction/python_quickstart
introduction/building
introduction/apis_and_wrappers
```

```{toctree}
:hidden:
:caption: API

api/python
api/cpp
api/configuration_files
```

```{toctree}
:hidden:
:caption: Environments

environments/default
environments/third_party
environments/creating_custom
```

```{toctree}
:hidden:
:caption: Other

faq/index
citation/index
Original website <https://vizdoom.cs.put.edu.pl/>
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/ViZDoom>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/ViZDoom/blob/master/docs/README.md>
```
