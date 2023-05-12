[![PyPI version](https://badge.fury.io/py/vizdoom.svg)](https://badge.fury.io/py/vizdoom) ![Build](https://github.com/mwydmuch/ViZDoom/workflows/Build/badge.svg) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/vizdoom-text.png" width="500px"/>
</p>

ViZDoom allows developing AI **bots that play Doom using only visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

![vizdoom_deadly_corridor](http://www.cs.put.poznan.pl/mkempka/misc/vizdoom_gifs/vizdoom_corridor_segmentation.gif)


## Features
- Multi-platform (Linux, macOS, Windows),
- API for Python and C++,
- [Gymnasium](https://gymnasium.farama.org/)/[OpenAI Gym](https://www.gymlibrary.dev/) environment wrappers (thanks to [Arjun KG](https://github.com/arjun-kg) [Benjamin Noah Beal](https://github.com/bebeal), [Lawrence Francis](https://github.com/ldfrancis), and [Mark Towers](https://github.com/pseudo-rnd-thoughts)),
- Easy-to-create custom scenarios (visual editors, scripting language, and examples available),
- Async and sync single-player and multiplayer modes,
- Fast (up to 7000 fps in sync mode, single-threaded),
- Lightweight (few MBs),
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision),
- Automatic labeling of game objects visible in the frame,
- Access to the audio buffer (thanks to [Shashank Hegde](https://github.com/hegde95)),
- Access to the list of actors/objects and map geometry,
- Off-screen rendering,
- Episodes recording,
- Time scaling in async mode.

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).

Julia (thanks to [Jun Tian](https://github.com/findmyway)), Lua, and Java bindings are available in other branches but are no longer maintained.


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
or

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

### Linux
Both x86-64 and ARM64 architectures are supported.
ViZDoom requires C++11 compiler, CMake 3.4+, Boost 1.65+ SDL2, OpenAL (optional) and Python 3.7+. Below you will find instructrion how to install these dependencies.

#### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

To install ViZDoom run (may take few minutes):
```
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev
pip install vizdoom
```
We recommend using at least Ubuntu 18.04+ or Debian 10+ with Python 3.7+.

#### dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)

To install ViZDoom run (may take few minutes):
```
dnf install cmake git boost-devel SDL2-devel openal-soft-devel
pip install vizdoom
```
We recommend using at least Fedora 35+ or RHEL/CentOS/Alma/Rocky Linux 9+ with Python 3.7+. To install openal-soft-devel on RHEL/CentOS/Alma/Rocky Linux 9, one needs to use `dnf --enablerepo=crb install`.

#### Conda-based installation
To install ViZDoom on a conda environment (no system-wide installations required):
```
conda install -c conda-forge boost cmake sdl2
git clone https://github.com/mwydmuch/ViZDoom.git --recurse-submodules
cd ViZDoom
python setup.py build && python setup.py install
```
Note that `pip install vizdoom` won't work with conda install and you have to follow these steps.


### macOS
Both Intel and Apple Silicon CPUs are supported.

To install ViZDoom on run (may take few minutes):
```
brew install cmake git boost openal-soft sdl2
pip install vizdoom
```
We recommend using at least macOS High Sierra 10.13+ with Python 3.7+.
On Apple Silicon (M1 and M2), make sure you are using Python for Apple Silicon.


### Windows
To install pre-build release for Windows 10 or 11 64-bit and Python 3.7+ just run (should take few seconds):
```
pip install vizdoom
```

Please note that the Windows version is not as well-tested as Linux and macOS versions. It can be used for development and testing if you want to conduct experiments on Windows, please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).


### Gymnasium/Gym wrappers
Gymnasium environments are installed along with ViZDoom.
See [documentation](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Gymnasium.md) and [examples](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/gymnasium_wrapper.py) on the use of Gymnasium API.

OpenAI-Gym wrappers are also available, to install them run:
```
pip install vizdoom[gym]
```
See [documentation](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Gym.md) and [examples](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/gym_wrapper.py) on the use of Gym API.
**OpenAI-Gym wrappers are deprecated and will be removed in future versions in favour of Gymnasium.**


## Examples

- [Python](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python) (contain learning examples implemented in PyTorch, TensorFlow and Theano)
- [C++](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/c%2B%2B)

Python examples are currently the richest, so we recommend to look at them, even if you plan to use other language. The API is almost identical for all languages.

**See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).**


## Original Doom graphics

Unfortunately, we cannot distribute ViZDoom with original Doom graphics.
If you own original Doom or Doom 2 games, you can replace [Freedoom](https://freedoom.github.io/) graphics by placing `doom.wad` or `doom2.wad` into your working directory or `vizdoom` package directory.

Alternatively, any base game WAD (including other Doom engine-based games and custom/community games) can be used by pointing to it with the [`set_doom_game_path/setDoomGamePath`](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/DoomGame.md#-setdoomscenariopath) method.


## Documentation

Detailed description of all types and methods:

- **[DoomGame](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/DoomGame.md)**
- **[Types](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Types.md)**
- [Configuration files](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/ConfigFile.md)
- [Exceptions](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Exceptions.md)
- [Utilities](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Utilities.md)

Additional documents:

- **[FAQ](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/FAQ.md)**
- [Changelog](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Changelog.md) for 1.1.X version.

Full documentation of the Doom engine and ACS scripting language can be found on
[ZDoom Wiki](https://zdoom.org/wiki/).

Useful articles:

- [ZDoom Wiki: ACS (scripting language)](https://zdoom.org/wiki/ACS)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs)
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs)


## Awesome Doom tools/projects

- [SLADE3](http://slade.mancubus.net/) - great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - another great Doom map editor for Windows.
- [OBLIGE](http://oblige.sourceforge.net/) - Doom random map generator and [PyOblige](https://github.com/mwydmuch/PyOblige) is a simple Python wrapper for it.
- [Omgifol](https://github.com/devinacker/omgifol) - nice Python library for manipulating Doom maps.
- [NavDoom](https://github.com/agiantwhale/navdoom) - Maze navigation generator for ViZDoom (similar to DeepMind Lab).
- [MazeExplorer](https://github.com/microsoft/MazeExplorer) - More sophisticated maze navigation generator for ViZDoom.
- [Sample Factory](https://github.com/alex-petrenko/sample-factory) - A high performance reinforcement learning framework for ViZDoom.
- [EnvPool](https://github.com/sail-sg/envpool/) - A high performance vectorized environment for ViZDoom.
- [Obsidian](https://github.com/dashodanger/Obsidian) - Doom random map generator, continuation of OBLIGE.


## Contributions

This project is maintained and developed in our free time. All bug fixes, new examples, scenarios and other contributions are welcome! We are also open to features ideas and design suggestions.


## License

Code original to ViZDoom is under MIT license. ZDoom uses code from several sources with [varying licensing schemes](http://zdoom.org/wiki/license).
