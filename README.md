[![PyPI version](https://badge.fury.io/py/vizdoom.svg)](https://badge.fury.io/py/vizdoom) [![Build and test](https://github.com/Farama-Foundation/ViZDoom/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/Farama-Foundation/ViZDoom/actions/workflows/build-and-test.yml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/docs/_static/img/vizdoom-text.png" width="500px"/>
</p>

ViZDoom allows developing AI **bots that play Doom using only visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://zdoom.org) engine to provide the game mechanics.

![ViZDoom Demo](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/docs/_static/img/vizdoom-demo.gif)


## Features
- API for Python (including [Gymnasium](https://gymnasium.farama.org/)/Gym wrappers) and C++,
- Multi-platform (Linux, macOS, Windows),
- Fast (up to 7000 frames/steps per second in sync mode, single-threaded on a modern CPU),
- Lightweight (few MBs),
- Easy-to-create custom scenarios (visual editors, scripting language, and examples available),
- Async and sync single-player and multiplayer modes,
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision),
- Automatic labeling and categorization of game objects visible in the frame,
- Access to the audio buffer,
- Access to the list of actors/objects and map geometry,
- Off-screen rendering,
- Episodes recording,
- In-game time scaling in async mode.

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).


## Cite as
> M Wydmuch, M Kempka & W Jaśkowski, ViZDoom Competitions: Playing Doom from Pixels, IEEE Transactions on Games, vol. 11, no. 3, pp. 248-259, 2019
([arXiv:1809.03470](https://arxiv.org/abs/1809.03470))
```
@article{Wydmuch2019ViZdoom,
  author  = {Marek Wydmuch and Micha{\l} Kempka and Wojciech Ja\'skowski},
  title   = {{ViZDoom} {C}ompetitions: {P}laying {D}oom from {P}ixels},
  journal = {IEEE Transactions on Games},
  year    = {2019},
  volume  = {11},
  number  = {3},
  pages   = {248--259},
  doi     = {10.1109/TG.2018.2877047},
  note    = {The 2022 IEEE Transactions on Games Outstanding Paper Award}
}
```
or/and

> M. Kempka, M. Wydmuch, G. Runc, J. Toczek & W. Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))
```
@inproceedings{Kempka2016ViZDoom,
  author    = {Micha{\l} Kempka and Marek Wydmuch and Grzegorz Runc and Jakub Toczek and Wojciech Ja\'skowski},
  title     = {{ViZDoom}: A {D}oom-based {AI} Research Platform for Visual Reinforcement Learning},
  booktitle = {IEEE Conference on Computational Intelligence and Games},
  year      = {2016},
  address   = {Santorini, Greece},
  month     = {Sep},
  pages     = {341--348},
  publisher = {IEEE},
  doi       = {10.1109/CIG.2016.7860433},
  note      = {The Best Paper Award}
}
```


## Python quick start

### Note about versions

#### ViZDoom 1.3.0.dev (development)

We are currently working on version 1.3.0, and decideded to make development releases from time to time on PyPI.
These releases are mostly incremental updates to the 1.2.4 version.
We aim to keep the API stable, so 1.3.0 development releases should be compatible with the 1.2.4 version.
However some minor behaviors may change, and bugs may be introduced.
The main changes include/will include:
- The update of Gymnasium environments to the latest Gymnasium API, and some minor improvements and bug fixes,
- New version of the old environments and completely new environments,
- New features including methods for easy setting of shaping reward, and natual language log of the game events,
- Python 3.13 support,
- Python type hinting support.

To install the latest development version of ViZDoom, just run:
```sh
pip install vizdoom --pre
```

#### ViZDoom 1.2.4 (stable)

ViZDoom 1.2.4 is the latest stable release, which is compatible with Python 3.8-3.12 and both NumPy 1.x and 2.x versions.
⚠️ Versions 1.2.3 and below do not work correctly with NumPy 2.0+. To use NumPy 2.0+ please upgrade ViZDoom to version 1.2.4+.
⚠️ For Python 3.13 install ViZDoom 1.3.0.dev release.


### Linux
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
Both x86-64 and AArch64 (ARM64) architectures are supported.
Wheels are available for Python 3.8+ on Linux.

⚠️ To use audio features, you need OpenAL install in your system.
On apt-based distros (Ubuntu, Debian, Linux Mint, etc.)
```sh
apt install libopenal-dev
```

On dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)
```sh
dnf install openal-soft-devel
```

If Python wheel is not available for your platform (Python version <3.8, distros below manylinux_2_28 standard), pip will try to install (build) ViZDoom from the source.
ViZDoom requires a C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional), and Python 3.8+ to install from source. See [documentation](https://vizdoom.farama.org/introduction/python_quickstart/) for more details.


### macOS
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
Both Intel and Apple Silicon CPUs are supported.
Pre-build wheels are available for Intel macOS 13.0+ and Apple Silicon (M-series chips) macOS 14.0+.


If Python wheel is not available for your platform (Python version <3.8, older macOS version), pip will try to install (build) ViZDoom from the source.
ViZDoom requires a C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional), and Python 3.8+ to install from source. See [documentation](https://vizdoom.farama.org/introduction/building/) for more details how to install dependencies.


### Windows
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
At the moment, only x86-64 architecture is supported on Windows.
Wheels are available for Python 3.9+ on Windows.

Please note that the Windows version is not as well-tested as Linux and macOS versions.
It can be used for development and testing but if you want to conduct serious (time and resource-extensive) experiments on Windows,
please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl) with Linux version.


### Gymnasium/Gym wrappers
Gymnasium environments are installed along with ViZDoom and are available on all platforms.
See [documentation](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Gymnasium.md) and [examples](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/gymnasium_wrapper.py) on the use of Gymnasium API.


## Examples
- [Python](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python) (contain learning examples implemented in PyTorch, TensorFlow, and Theano)
- [C++](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/c%2B%2B)

Python examples are currently the richest, so we recommend looking at them, even if you plan to use C++.
The API is almost identical between the languages, with the only difference being that Python uses snake_case and C++ camelCase for methods and functions.


## Original Doom graphics
Unfortunately, we cannot distribute ViZDoom with original Doom graphics.
If you own original Doom or Doom 2 games, you can replace [Freedoom](https://freedoom.github.io/) graphics by placing `doom.wad` or `doom2.wad` into your working directory or `vizdoom` package directory.

Alternatively, any base game WAD (including other Doom engine-based games and custom/community games) can be used by pointing to it with the [`set_doom_game_path/setDoomGamePath`](https://vizdoom.farama.org/main/api/python/doom_game/index.html#vizdoom.DoomGame.set_doom_game_path) method.


## Documentation
Detailed descriptions of all ViZDoom types and methods can be found in the [documentation](https://vizdoom.farama.org/).

Full documentation of the ZDoom engine and ACS scripting language can be found on
[ZDoom Wiki](https://zdoom.org/wiki/).

Useful articles (for advanced users who want to create custom environments/scenarios):
- [ZDoom Wiki: ACS (scripting language)](https://zdoom.org/wiki/ACS)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs)
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs)


## Awesome Doom tools/projects
- [SLADE3](http://slade.mancubus.net/) - Great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - Another great Doom map editor for Windows.
- [OBLIGE](http://oblige.sourceforge.net/) - Doom random map generator and [PyOblige](https://github.com/mwydmuch/PyOblige) is a simple Python wrapper for it.
- [Omgifol](https://github.com/devinacker/omgifol) - Nice Python library for manipulating Doom maps.
- [NavDoom](https://github.com/agiantwhale/navdoom) - Maze navigation generator for ViZDoom (similar to DeepMind Lab).
- [MazeExplorer](https://github.com/microsoft/MazeExplorer) - A more sophisticated maze navigation generator for ViZDoom.
- [Sample Factory](https://github.com/alex-petrenko/sample-factory) - A high-performance reinforcement learning framework for ViZDoom.
- [EnvPool](https://github.com/sail-sg/envpool/) - A high-performance vectorized environment for ViZDoom.
- [Obsidian](https://github.com/dashodanger/Obsidian) - Doom random map generator, a continuation of OBLIGE.
- [LevDoom](https://github.com/TTomilin/LevDoom) - Generalization benchmark in ViZDoom featuring difficulty levels in visual complexity.
- [COOM](https://github.com/TTomilin/COOM) - Continual learning benchmark in ViZDoom offering task sequences with diverse objectives.
- [HASARD](https://github.com/TTomilin/HASARD) - A safe reinforcement learning benchmark in ViZDoom

If you have a cool project that uses ViZDoom or could be interesting to ViZDoom community, feel free to open PR to add it to this list!


## Contributions
This project is maintained and developed in our free time. All bug fixes, new examples, scenarios, and other contributions are welcome! We are also open to feature ideas and design suggestions.

We have a roadmap for future development work for ViZDoom available [here](https://github.com/Farama-Foundation/ViZDoom/issues/546).


## License
The code original to ViZDoom is under MIT license. ZDoom uses code from several sources with [varying licensing schemes](http://zdoom.org/wiki/license).
