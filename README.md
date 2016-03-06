#ViZDoom

Doom-based environment for visual learning. ViZDoom uses [Zdoom engine]( https://github.com/rheit/zdoom) to provide the game mechanics.

---
## Supported Languages:
* Python 2
* Java
* C++

---
## Building
Compilation for Windows is discouraged and downloading already [compiled version](http://www.cs.put.poznan.pl/visualdoomai/TOBEGIVENLATER) is encouraged.

###Prerequisites:
* Linux
* cmake 2.4+
* make
* gcc 4.0+
* Boost (system, filesystem and thread)
* Python 2.6+ with Numpy and Boost.Python for Pyhon binding (optional)
* Java compiler and Java for Java binding (optional)

Additionaly [Zdoom dependancies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

###Building commands
```bash
cmake -DCMAKE_BUILD_TYPE=Release
make
```
####Options
Java and Python bindings are optional, to build them additional cmake flags are necessary (they can be combined).
#####Java Binding
```bash
-DBUILD_JAVA=ON
```
#####Python Binding
```bash
-DBUILD_PYTHON=ON
```
---
##Examples

To run provided examples [freedoom2.wad]( https://freedoom.github.io/download.html) file is needed and should be placed in scenarios subdirectory.

###Python
To check examples in Python go to [examples/python](https://github.com/Marqt/ViZDoom/tree/master/examples/python) and read README file.
###C++
 TODO
###Java
 TODO
