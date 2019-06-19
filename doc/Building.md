# Installation/Building

* [Dependencies](#deps)
  * [Linux](#linux_deps)
  * [MacOS](#macos_deps)
  * [Windows](#windows_deps)
* [Installation via PyPI(pip/conda)](#pypi)
* [Installation via LuaRocks](#luarocks)
* [Installation of build Windows binaries](#windows_bin)
* [Building (not recommended)](#build)
  * [Linux](#linux_build)
  * [MacOS](#macos_build)
  * [Windows](#windows_build)
  * [Compilation output](#output)
  * [Manual installation](#manual-install)


## <a name="deps"></a> Dependencies

Even if you plan to install ViZDoom via PyPI or LuaRocks, you need to install some dependencies in your system first.


### <a name="linux_deps"></a> Linux
* CMake 2.8+
* Make
* GCC 4.9+
* Boost libraries 1.54+
* Python 2.7+ or Python 3+ with Numpy for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.


To get all dependencies (except JDK) on Ubuntu execute the following commands in the shell (requires root access):
```bash
# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 2 dependencies
sudo apt-get install python-dev python-pip
pip install numpy
# or install Anaconda 2 and add it to PATH

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Lua binding dependencies
sudo apt-get install liblua5.1-dev
# Lua shipped with Torch can be used instead, so it isn't needed if installing via LuaRocks

# Julia dependencies
sudo apt-get install julia
julia
julia> using Pkg
julia> Pkg.add("CxxWrap")
```

To get Torch see: [Getting started with Torch](http://torch.ch/docs/getting-started.html).


### <a name="macos_deps"></a> MacOS
* CMake 2.8+
* Clang 3.4+
* Boost libraries
* Python 2.7+ or Python 3+ with Numpy for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get dependencies install [homebrew](https://brew.sh/)

```sh
# ZDoom dependencies and Boost libraries
brew install cmake boost sdl2 wget

# Python 2 dependencies
brew install python
pip install numpy

# Python 3 dependencies
brew install python3
pip3 install numpy
# or install

# Julia dependencies
brew cask install julia
julia
julia> Pkg.add("CxxWrap")
```

To get Torch see: [Getting started with Torch](http://torch.ch/docs/getting-started.html).


### <a name="windows_deps"></a> Windows
* CMake 2.8+
* Visual Studio 2012+
* Boost libraries
* Python 2.7+ or Python 3.4+ with Numpy for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 or LUA 5.1 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).


## <a name="pypi"></a> Installation via PyPI (recommended for Python users)

ViZDoom for Python can be installed via **pip/conda** on Linux and MacOS, and it is strongly recommended.
However you will still need to install **[Linux](#linux_deps)/[MacOS](#macos_deps) dependencies**.

> Pip installation is not supported on Windows at the moment, but we hope some day it will.

To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
# use pip3 for Python3
sudo pip install vizdoom
```

To install the newest version from the repository:
```bash
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
# use pip3 for Python3
sudo pip install .
```

Or without cloning it yourself:
```bash
# use pip3 for Python3
sudo pip install git+https://github.com/mwydmuch/ViZDoom
```


## <a name="luarocks"></a> Installation via LuaRocks (recommended for Torch7 users)

ViZDoom for Torch7 can be installed via **luarocks** on Linux and MacOS, and it is strongly recommended.
However you will still need to install **[Linux](#linux_deps)/[MacOS](#macos_deps) dependencies**.

To install the most stable official release from [LuaRocks](https://pypi.python.org/pypi):
```bash
luarocks install vizdoom
```

To install the newest version from the repository:
```bash
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
luarocks make
```


## <a name="windows_bin"></a> Installation of Windows binaries

For Windows we are providing a compiled environment that can be download from [releases](https://github.com/mwydmuch/ViZDoom/releases) page.
To install it for Python, copy files to `site-packages` folder.

Location of `site-packages` depends on Python distribution:
- Python: `python_root\Lib\site-packges`
- Anaconda: `anaconda_root\lib\pythonX.X\site-packages`


## <a name="build"></a> Building

### <a name="linux_build"></a> Linux

>>> Using [pip/conda](#pypi) is the recommended way to install ViZDoom, please try it first unless you are sure you want to compile the package by hand.

In ViZDoom's root directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON -DBUILD_JULIA=ON
make
```

where `-DBUILD_PYTHON=ON`, `-DBUILD_JAVA=ON`, `-DBUILD_LUA=ON` and `-DBUILD_JULIA=ON` CMake options for Python, Java, Lua and Julia bindings are optional (default OFF). To force building bindings for Python3 instead of the first version found use `-DBUILD_PYTHON3=ON`.

To build Julia binding you first need to install CxxWrap package by running `julia` and using `Pkg.add("CxxWrap")` command (see [Linux dependencies](#linux_deps)). Then you need to manually set `JlCxx_DIR` variable:

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_JULIA=ON \
-DJlCxx_DIR=~/.julia/vX.X/CxxWrap/deps/usr/lib/cmake/JlCxx/
```


### <a name="macos_build"></a> MacOS

>>> Using [pip/conda](#pypi) is the recommended way to install ViZDoom, please try it first unless you are sure you want to compile the package by hand.

Run CMake and build generated Makefile.

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON -DBUILD_JULIA=ON
make
```

where `-DBUILD_PYTHON=ON`, `-DBUILD_JAVA=ON`, `-DBUILD_LUA=ON` and `-DBUILD_JULIA=ON` CMake options for Python, Java, Lua and Julia bindings are optional (default OFF). To force building bindings for Python3 instead of the first version found use `-DBUILD_PYTHON3=ON`.

Users with brew-installed Python/Anaconda **may** need to manually set `PYTHON_EXECUTABLE`, `PYTHON_INCLUDE_DIR`, `PYTHON_LIBRARY` variables:

It should look like this for brew-installed Python (use `-DBUILD_PYTHON3=ON`, `include/pythonX.Xm` and `lib/libpythonX.Xm.dylib` for Python 3):

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \
-DPYTHON_EXECUTABLE=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/bin/pythonX \
-DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/include/pythonX.X \
-DPYTHON_LIBRARY=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/libpythonX.X.dylib \
-DNUMPY_INCLUDES=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/pythonX.X/site-packages/numpy/core/include
```

Or for Anaconda (use `-DBUILD_PYTHON3=ON`, `include/pythonX.Xm` and `lib/libpythonX.Xm.dylib` for Python 3):

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \
-DPYTHON_EXECUTABLE=~/anacondaX/bin/pythonX \
-DPYTHON_INCLUDE_DIR=~/anacondaX/include/pythonX.X \
-DPYTHON_LIBRARY=~/anacondaX/lib/libpythonX.X.dylib \
-DNUMPY_INCLUDES=~/anacondaX/lib/pythonX.X/site-packages/numpy/core/include
```

To build Julia binding, you first need to install CxxWrap package by running `julia` and using `Pkg.add("CxxWrap")` command (see [MacOS dependencies](#macos_deps)). Then you need to manually set `JlCxx_DIR` variable:

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_JULIA=ON \
-DJlCxx_DIR=~/.julia/vX.X/CxxWrap/deps/usr/lib/cmake/JlCxx/
```

### <a name="windows_build"></a> Windows

Setting up the compilation on Windows is really tedious so using the [precompiled binaries](#windows_bin) is recommended.

`vizdoom` directory from Python builds contains complete Python package for Windows.
You can copy it to your project directory or copy it into `python_dir/lib/site-packages/vizdoom` to install it globally in your system.


Run CMake GUI, select ViZDoom root directory and set paths to:
* BOOST_ROOT
* BOOST_INCLUDEDIR
* BOOST_LIBRARYDIR
* PYTHON_INCLUDE_DIR (optional, for Python/Anaconda bindings)
* PYTHON_LIBRARY (optional, for Python/Anaconda bindings)
* NUMPY_INCLUDES (optional, for Python/Anaconda bindings)
* LUA_LIBRARIES (optional, for Lua/Torch bindings)
* LUA_INCLUDE_DIR (optional, for Lua/Torch bindings)
* ZDoom dependencies paths

In configuration select BUILD_PYTHON, BUILD_PYTHON3 and BUILD_JAVA options for Python and Java bindings (optional, default OFF).

Use generated Visual Studio solution to build all parts of ViZDoom environment.


### <a name="output"></a> Compilation output
Compilation output will be placed in `vizdoom_root_dir/bin` and it should contain following files.

* `bin/vizdoom / vizdoom.exe` - ViZDoom executable
* `bin/vizdoom.pk3` - resources file used by ViZDoom (needed by ViZDoom executable)
* `bin/libvizdoom.a / vizdoom.lib` - C++ ViZDoom static library
* `bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib` -  C++ ViZDoom dynamically linked library
* `bin/python2/vizdoom.so / vizdoom.pyd / vizdoom.dylib` - ViZDoom Python 2 module
* `bin/pythonX.X/vizdoom.so / vizdoom.pyd ` - ViZDoom Python X.X module
* `bin/pythonX.X/pip_package` - complete ViZDoom Python X.X package
* `bin/lua/vizdoom.so / vizdoom.so / vizdoom.dylib` - ViZDoom Lua C module
* `bin/lua/luarocks_package` - complete ViZDoom Torch package
* `bin/java/libvizdoom.so / vizdoom.dll / libvizdoom.dylib` -  ViZDoom library for Java
* `bin/java/vizdoom.jar` -  Contains ViZDoom Java classes


### <a name="manual-install"></a> Manual installation

To manually install Python package copy `vizdoom_root_dir/bin/pythonX.X/pip_package` contents to `python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom`.

To manually install Torch package copy `vizdoom_root_dir/bin/lua/luarocks_package` contents to `torch_root_dir/install/lib/lua/5.1/vizdoom` and `vizdoom_root_dir/bin/lua/luarocks_shared_package` contents to `torch_root_dir/install/share/lua/5.1/vizdoom`.
