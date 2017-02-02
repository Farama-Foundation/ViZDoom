#ViZDoom
[http://vizdoom.cs.put.edu.pl](http://vizdoom.cs.put.edu.pl)

ViZDoom allows developing AI **bots that play Doom using only the visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

## Features
- Multi-platform,
- API for C++, Lua, Java and Python,
- Easy-to-create custom scenarios (examples available),
- Async and sync single-player and multi-player modes,
- Fast (up to 7000 fps in sync mode, single threaded),
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision)
- Automatic labeling game objects visible in the frame
- Off-screen rendering,
- Episodes recording,
- Time scaling in async mode,
- Lightweight (few MBs).

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).


## Cite as

>Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek & Wojciech Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))
### Bibtex:
```
@inproceedings{Kempka2016ViZDoom,
  author    = {Micha{\l} Kempka and Marek Wydmuch and Grzegorz Runc and Jakub Toczek and Wojciech Ja\'skowski},
  title     = {{ViZDoom}: A {D}oom-based {AI} Research Platform for Visual Reinforcement Learning},
  booktitle = {IEEE Conference on Computational Intelligence and Games},  
  year      = {2016},
  url       = {http://arxiv.org/abs/1605.02097},
  address   = {Santorini, Greece},
  Month     = {Sep},
  Pages     = {341--348},
  Publisher = {IEEE},
  Note      = {The best paper award}
}
```

## Documentation

Detailed description of all types and methods:

- **[DoomGame](DoomGame.md)**
- **[Types](Types.md)**
- [Configuration files](ConfigFile.md)
- [Exceptions](Exceptions.md)
- [Utilities](Utilities.md)

[Changelog](Changelog.md) for 1.1.0 version.
