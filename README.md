#ViZDoom [![Build Status](https://travis-ci.org/Marqt/ViZDoom.svg?branch=master)](https://travis-ci.org/Marqt/ViZDoom)
[http://vizdoom.cs.put.edu.pl](http://vizdoom.cs.put.edu.pl)

ViZDoom allows developing AI **bots that play Doom using only the visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

## Features
* API for C++, Python and Java,
* Easy-to-create custom scenarios (examples available),
* Single-player (sync and async) and multi-player (async) modes,
* Fast (up to 7000 fps in sync mode, single threaded),
* Customizable resolution and rendering parameters,
* Access to the depth buffer (3D vision)
* Off-screen rendering,
* Lightweight (few MBs),
* Multi-platform

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).

## Planned Features (June)
* Lua bindings,
* Multi-player working in sync mode,
* Labeling game objects visible in the frame,
* Time scaling in async mode.

## Cite as

>Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek & Wojciech Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, 2016	([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))

---
## Building

###Linux

####Dependencies
* CMake 3.0+
* Make
* GCC 4.6+
* Boost libraries
* Python 2.7+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

####Compiling
In ViZDoom's root directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON
make
```

``-DBUILD_PYTHON=ON`` and ``-DBUILD_JAVA=ON`` CMake options for Python and Java bindings are optional (default OFF).

###Windows

We are providing compiled runtime binaries and development libraries for Windows [here](https://github.com/Marqt/ViZDoom/releases/download/1.0.1/ViZDoom-1.0.2-Win-x86_64.zip).

####Dependencies
* CMake 3.0+
* Visual Studio 2012+
* Boost libraries
* Python 2.7+ with Numpy and Boost.Python for Python binding (optional)
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

In configuration select BUILD_PYTHON and BUILD_JAVA options for Python and Java bindings (optional, default OFF).

Use generated Visual Studio solution to build all ViZDoom's parts.

###OSX
Untested, code is compatible, CMake still may need minor adjustments.
Let us know if You are using ViZDoom on OSX.

####Dependencies
* CMake 3.0+
* XCode 5+
* Boost libraries
* Python 2.7+ with Numpy and Boost.Python for Python binding (optional)
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
* ``bin/java/libvizdoom.so (vizdoom.dll)`` -  ViZDoom library for Java
* ``bin/java/vizdoom.jar`` -  Contains ViZDoom Java classes

## Docker

* [Dockerfile](https://github.com/maciejjaskowski/ViZDoom-docker)
* [Docker image](https://hub.docker.com/r/mjaskowski/vizdoom/)

Note: third-party maintained

---
##Examples

Before running the provided examples, make sure that [freedoom2.wad](https://freedoom.github.io/download.html) is placed it in the ``scenarios`` subdirectory (on Linux it should be done automatically by the building process).

* [Python](https://github.com/Marqt/ViZDoom/tree/master/examples/python)
* [C++](https://github.com/Marqt/ViZDoom/tree/master/examples/c%2B%2B)
* [Java](https://github.com/Marqt/ViZDoom/tree/master/examples/java)

See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).

---
##License

Code original to ViZDoom is under MIT license. ZDoom uses code from several sources which varied licensing schemes, more informations [here](http://zdoom.org/wiki/license).
