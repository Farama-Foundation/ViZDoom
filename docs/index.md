---
hide-toc: true
firstpage:
lastpage:
---

# ViZDoom allows developing AI bots that play Doom using only the visual information.

```{figure} _static/REPLACE_ME.gif
   :alt: ViZDoom
   :width: 500
```

**Basic example:**

```{code-block} python

import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

```{toctree}
:hidden:
:caption: Introduction

introduction/quickstart
introduction/installation
introduction/wrappers
```

```{toctree}
:hidden:
:caption: C++ API

api_cpp/doomGame
api_cpp/types
api_cpp/configurationFiles
api_cpp/exceptions
api_cpp/utils
```

```{toctree}
:hidden:
:caption: Others

faq/index
citation/index
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/ViZDoom>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/ViZDoom/blob/master/docs/README.md>
```
