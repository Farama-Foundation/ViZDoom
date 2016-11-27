#ViZDoom [![Build Status](https://travis-ci.org/Marqt/ViZDoom.svg?branch=master)](https://travis-ci.org/Marqt/ViZDoom)
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

## Windows build
For Windows we are providing compiled runtime binaries and development libraries:

### [1.1.0rc](https://github.com/Marqt/ViZDoom/releases/tag/1.1.0rc) (2016-11-26):
- [Python 2.7 (64-bit)](https://github.com/Marqt/ViZDoom/releases/download/1.1.0rca/ViZDoom-1.1.0rc-Win-Python27-x86_64.zip)
- [Python 3.4 (64-bit)](https://github.com/Marqt/ViZDoom/releases/download/1.1.0rca/ViZDoom-1.1.0rc-Win-Python34-x86_64.zip)
- [Python 3.5 (64-bit)](https://github.com/Marqt/ViZDoom/releases/download/1.1.0rca/ViZDoom-1.1.0rc-Win-Python35-x86_64.zip)
- [Lua51 & LuaJIT (64-bit)](https://github.com/Marqt/ViZDoom/releases/download/1.1.0rca/ViZDoom-1.1.0rc-Win-Lua51-LuaJIT-x86_64.zip)
- [Java (64-bit)](https://github.com/Marqt/ViZDoom/releases/download/1.1.0rca/ViZDoom-1.1.0rc-Win-Java-x86_64.zip)


## Examples

Before running the provided examples, make sure that [freedoom2.wad](https://freedoom.github.io/download.html) is placed it in the ``scenarios`` subdirectory (on Linux it should be done automatically by the building process):

- [Python](examples/python)
- [C++](examples/c%2B%2B)
- [Lua](examples/lua)
- [Java](examples/java)

Python examples are currently the richest, so we recommend to look at them, even if you plan to use other language.

**See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).**


## Building

- [Linux](doc/Building.md#linux)
- [Windows](doc/Building.md#windows)
- [OSX](doc/Building.md#osx)


## Documentation

Detailed description of all types and methods:

- **[DoomGame](doc/DoomGame.md)**
- **[Types](doc/Types.md)**
- [Configuration files](doc/ConfigFile.md)
- [Exceptions](doc/Exceptions.md)(soon)
- [Utilities](doc/Utilities.md)

[Changelog](doc/Changelog.md) for 1.1.0 version.


## License

Code original to ViZDoom is under MIT license. ZDoom uses code from several sources with [varying licensing schemes](http://zdoom.org/wiki/license).
