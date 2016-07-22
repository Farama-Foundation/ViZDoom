#ViZDoom [![Build Status](https://travis-ci.org/Marqt/ViZDoom.svg?branch=master)](https://travis-ci.org/Marqt/ViZDoom)
[http://vizdoom.cs.put.edu.pl](http://vizdoom.cs.put.edu.pl)

ViZDoom allows developing AI **bots that play Doom using only the visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

## Features
* Multi-platform,
* API for C++, Python and Java,
* Easy-to-create custom scenarios (examples available),
* Single-player (sync and async) and multi-player (async) modes,
* Fast (up to 7000 fps in sync mode, single threaded),
* Customizable resolution and rendering parameters,
* Access to the depth buffer (3D vision)
* Off-screen rendering,
* Episodes recording,
* Time scaling in async mode,
* Lightweight (few MBs).

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).

For the new features:
* Automatic labeling of game objects visible in the frame,
* Access to the top down map buffer.

Check out [1.1-dev](https://github.com/Marqt/ViZDoom/tree/1.1-dev) branch.

## Cite as

>Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek & Wojciech Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, Proceedings of Computational Intelligence in Games Conference, Santorini, Greece, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))

--
##Examples

Before running the provided examples, make sure that [freedoom2.wad](https://freedoom.github.io/download.html) is placed it in the ``scenarios`` subdirectory (on Linux it should be done automatically by the building process):

* [Python](examples/python) 
* [C++](examples/c%2B%2B)
* [Java](examples/java)

Python examples are currently the richest, so we recommend to look at them, even if you plan to use C++ or Java.

See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).

## Documentation
Apart from the [examples](examples) and the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial), the most complete source of information about the ViZDoom API can be found in a [bachelor thesis](http://www.cs.put.poznan.pl/wjaskowski/pub/theses/ViZDoom_BScThesis.pdf), which describes the initial version of this project (note, however, that it is not entirely up-to-date).

--
## Building

###Linux

####Dependencies
* CMake 3.0+
* Make
* GCC 4.6+
* Boost libraries (tested on 1.54, 1.58, 1.59, 1.61)
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

####Compiling
In ViZDoom's root directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON
make
```

``-DBUILD_PYTHON=ON`` and ``-DBUILD_JAVA=ON`` CMake options for Python and Java bindings are optional (default OFF). To force building bindings for Python3 instead of first version found use ``-DBUILD_PYTHON3=ON`` (needs Boost.Python builded with Python 3, default OFF).

###Windows

We are providing compiled runtime binaries and development libraries for Windows:
[1.0.4](https://github.com/Marqt/ViZDoom/releases/download/1.0.4/ViZDoom-1.0.4-Win-x86_64.zip) or [1.1.0pre](https://github.com/Marqt/ViZDoom/releases/download/1.1.0pre-CIG2016-warm-up-fixed/ViZDoom-1.1.0pre-CIG2016-warm-up-Win-x86_64.zip).

####Dependencies
* CMake 3.0+
* Visual Studio 2012+
* Boost libraries
* Python 2.7+ or Python 3.4+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.

####Compiling
Run CMake GUI, select ViZDoom's root directory and set paths to:
* BOOST_ROOT
* BOOST_INCLUDEDIR
* BOOST_LIBRARYDIR
* PYTHON_INCLUDE_DIR (optional)
* PYTHON_LIBRARY (optional)
* NUMPY_INCLUDES (optional)
* ZDoom dependencies paths

In configuration select BUILD_PYTHON, BUILD_PYTHON3 and BUILD_JAVA options for Python and Java bindings (optional, default OFF).

Use generated Visual Studio solution to build all ViZDoom's parts.

###OSX
Untested, code is compatible, CMake still may need minor adjustments.
Let us know if You are using ViZDoom on OSX.

####Dependencies
* CMake 3.0+
* XCode 5+
* Boost libraries
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

####Compiling
Run CMake and use generated project.

Users with brew-installed Python may need to manually set:
``-DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.x.x/Frameworks/Python.framework/Versions/2.7/include/python2.7`` and 
``-DPYTHON_LIBRARY=/usr/local/Cellar/python/2.x.x/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib``

####Configuration
Craeting symlink to the app executable may be need:
``rm bin/vizdoom && ln -s vizdoom.app/Contents/MacOS/vizdoom bin/vizdoom``

###Compilation output
Compilation output will be placed in ``vizdoom_root_dir/bin`` and it should contain following files (Windows names are in brackets):

* ``bin/vizdoom (vizdoom.exe)`` - ViZDoom executable
* ``bin/vizdoom.pk3`` - resources file used by ViZDoom (needed by ViZDoom executable)
* ``bin/libvizdoom.a (vizdoom.lib)`` - C++ ViZDoom static library
* ``bin/libvizdoom.so (vizdoom.dll)`` -  C++ ViZDoom dynamically linked library
* ``bin/python/vizdoom.so (vizdoom.pyd)`` - ViZDoom Python module
* ``bin/python3/vizdoom.so (vizdoom.pyd)`` - ViZDoom Python3 module
* ``bin/java/libvizdoom.so (vizdoom.dll)`` -  ViZDoom library for Java
* ``bin/java/vizdoom.jar`` -  Contains ViZDoom Java classes

## Docker(outdated)

* [Dockerfile](https://github.com/maciejjaskowski/ViZDoom-docker)
* [Docker image](https://hub.docker.com/r/mjaskowski/vizdoom/)

Note: third-party maintained

---
##License

Code original to ViZDoom is under MIT license. ZDoom uses code from several sources which [varied licensing schemes](http://zdoom.org/wiki/license).
