#ViZDoom

ViZDoom allows to develop AI bots that play Doom using only the visual information (the screen buffer). It is primarly intended for research in machine visual learning, and reinforcement deep learning, in particular.

ViZDoom is based on [ZDoom]( https://github.com/rheit/zdoom) to provide the game mechanics.

## Features
* API for C++, Python and Java,
* Easy-to-create custom scenarios (examples available),
* Fast (>2000 frames per second),
* Single-player (sync or async) and multi-player modes,
* Customizable resolution and rendering parameters,
* Off-screen rendering,
* Lightweight (few MBs),
* Supports Linux and Windows.

ViZDoom API is reinforcement learning friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).

---
## Building

###Prerequisites
* cmake 2.4+
* make
* gcc 4.0+
* Boost (system, filesystem and thread)
* Python 2.6+ with Numpy and Boost.Python for Pyhon binding (optional)
* Java compiler and Java for Java binding (optional)

Additionaly, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

###Compiled versions
We provide a compiled [version for Windows](http://www.cs.put.poznan.pl/visualdoomai/TOBEGIVENLATER).

###Building commands
```bash
cmake -DCMAKE_BUILD_TYPE=Release
make
```
####Options
Java and Python bindings are optional; to build them, appropriate cmake flags have to be set.
#####Java Binding
```bash
-DBUILD_JAVA=ON
```
#####Python Binding
```bash
-DBUILD_PYTHON=ON
```

###Building on Windows
``TODO``

---
##Examples

To run the provided examples, download [freedoom2.wad]( https://freedoom.github.io/download.html) and place it in the ``scenarios`` subdirectory.

* [Python](https://github.com/Marqt/ViZDoom/tree/master/examples/python).
* [C++](https://github.com/Marqt/ViZDoom/tree/master/examples/c%2B%2B)
* [Java](https://github.com/Marqt/ViZDoom/tree/master/examples/java)
