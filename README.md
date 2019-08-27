# ViZDoom [![Build Status](https://travis-ci.org/mwydmuch/ViZDoom.svg?branch=master)](https://travis-ci.org/mwydmuch/ViZDoom)
[http://vizdoom.cs.put.edu.pl](http://vizdoom.cs.put.edu.pl)

ViZDoom allows developing AI **bots that play Doom using only the visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

![vizdoom_deadly_corridor](http://www.cs.put.poznan.pl/mkempka/misc/vizdoom_gifs/vizdoom_corridor.gif)


## Features
- Multi-platform,
- API for Python, C++, Lua, Java and Julia (thanks to [Jun Tian](https://github.com/findmyway)),
- Easy-to-create custom scenarios (visual editors, scripting language and examples available),
- Async and sync single-player and multi-player modes,
- Fast (up to 7000 fps in sync mode, single threaded),
- Lightweight (few MBs),
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision),
- Automatic labeling game objects visible in the frame,
- Off-screen rendering,
- Episodes recording,
- Time scaling in async mode.

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).


## Cite as
> M Wydmuch, M Kempka & W Jaśkowski, ViZDoom Competitions: Playing Doom from Pixels, IEEE Transactions on Games, in print,
[arXiv:1809.03470](https://arxiv.org/abs/1809.03470)
```
@article{wydmuch2018vizdoom,
  title={ViZDoom Competitions: Playing Doom from Pixels},
  author={Wydmuch, Marek and Kempka, Micha{\l} and Ja{\'s}kowski, Wojciech},
  journal={IEEE Transactions on Games},
  year={2018},
  publisher={IEEE}
}
```
or

> M. Kempka, M. Wydmuch, G. Runc, J. Toczek & W. Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))
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

## Installation/Building instructions

- **[PyPI (pip)](doc/Building.md#pypi)**
- **[LuaRocks](doc/Building.md#luarocks)**
- [Linux](doc/Building.md#linux_build)
- [MacOS](doc/Building.md#macos_build)
- [Windows](doc/Building.md#windows_build)


## Windows build
For Windows we are providing compiled runtime binaries and development libraries:

### [1.1.8pre](https://github.com/mwydmuch/ViZDoom/releases/tag/1.1.8pre) (2019-08-28):
- [Python 2.7 (64-bit)](https://github.com/mwydmuch/ViZDoom/releases/download/1.1.8pre/ViZDoom-1.1.8pre-Win-Python27-x86_64.zip)
- [Python 3.5 (64-bit)](https://github.com/mwydmuch/ViZDoom/releases/download/1.1.8pre/ViZDoom-1.1.8pre-Win-Python35-x86_64.zip)
- [Python 3.6 (64-bit)](https://github.com/mwydmuch/ViZDoom/releases/download/1.1.8pre/ViZDoom-1.1.8pre-Win-Python36-x86_64.zip)
- [Python 3.7 (64-bit)](https://github.com/mwydmuch/ViZDoom/releases/download/1.1.8pre/ViZDoom-1.1.8pre-Win-Python37-x86_64.zip)

See **[Installation of Windows binaries](doc/Building.md#windows_bin)**


## Examples

Before running the provided examples, make sure that [freedoom2.wad](https://freedoom.github.io/download.html) is placed in the same directory as the ViZDoom executable (on Linux and macOS it should be done automatically by the building process):

- [Python](examples/python) (contain learning examples implemented in PyTorch, TensorFlow and Theano)
- [C++](examples/c%2B%2B)
- [Lua](examples/lua) (contain learning example implemented in Torch)
- [Java](examples/java)
- [Julia](examples/julia)

Python examples are currently the richest, so we recommend to look at them, even if you plan to use other language. API is almost identical for all languages.

**See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).**


## Documentation

Detailed description of all types and methods:

- **[DoomGame](doc/DoomGame.md)**
- **[Types](doc/Types.md)**
- [Configuration files](doc/ConfigFile.md)
- [Exceptions](doc/Exceptions.md)
- [Utilities](doc/Utilities.md)

Additional documents:

- **[FAQ](doc/FAQ.md)**
- [Changelog](doc/Changelog.md) for 1.1.X version.

Also full documentation of engine and ACS scripting language can be found on
[ZDoom Wiki](https://zdoom.org/wiki/).

Useful parts:

- [ZDoom Wiki: ACS (scripting language)](https://zdoom.org/wiki/ACS)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs) 
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs) 


## Awesome Doom tools

- [SLADE3](http://slade.mancubus.net/) - great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - another great Doom map editor for Windows.
- [OBLIGE](http://oblige.sourceforge.net/) - Doom random map generator and [PyOblige](https://github.com/mwydmuch/PyOblige) is a simple Python wrapper for it.
- [Omgifol](https://github.com/devinacker/omgifol) - nice Python library for manipulating Doom maps.
- [NavDoom](https://github.com/agiantwhale/navdoom) - Maze navigation generator for ViZDoom (similar to DeepMind Lab).
- [MazeExplorer](https://github.com/microsoft/MazeExplorer) - More sophisticated maze navigation generator for ViZDoom. 

## Contributions

This project is maintained and developed in our free time. All bug fixes, new examples, scenarios and other contributions are welcome! We are also open to features ideas and design suggestions.


## License

Code original to ViZDoom is under MIT license. ZDoom uses code from several sources with [varying licensing schemes](http://zdoom.org/wiki/license).
