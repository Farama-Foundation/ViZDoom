# Installation/Building

* [Dependencies](#deps)
  * [Linux](#linux_deps)
  * [MacOS](#macos_deps)
  * [Windows](#windows_deps)
* [Installation via PyPI(pip/conda)](#pypi)
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
* CMake 3.1+
* Make
* GCC 6.0+
* Boost libraries 1.65.0+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

To get all dependencies on Ubuntu (we recommend using Ubuntu 18.04+) execute the following commands in the shell (requires root access). `scripts/linux_check_dependencies.sh` installs these for Python3:
```bash
# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Julia dependencies
sudo apt-get install julia
julia
julia> using Pkg
julia> Pkg.add("CxxWrap")
```

If you do not have a root access, you can use a conda (e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html)) environment to install dependencies to your environment only:
```
conda install -c conda-forge boost cmake gtk2 sdl2
```

Note that to install ViZDoom in a conda environment you have to pull, build and install ViZDoom manually with
```
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
python setup.py build && python setup.py install
```


### <a name="macos_deps"></a> MacOS
* CMake 3.1+
* Clang 5.0+
* Boost libraries 1.65.0+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get dependencies install [homebrew](https://brew.sh/)

```sh
# ZDoom dependencies and Boost libraries
brew install cmake boost openal-soft sdl2

# Python 3 dependencies
brew install python3
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Julia dependencies
brew cask install julia
julia
julia> using Pkg
julia> Pkg.add("CxxWrap")
```


### <a name="windows_deps"></a> Windows
* CMake 3.1+
* Visual Studio 2012+
* Boost 1.65+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).


## <a name="pypi"></a> Installation via PyPI (recommended for Python users)

ViZDoom for Python can be installed via **pip/conda** on Linux and MacOS, and it is strongly recommended.
However you will still need to install **[Linux](#linux_deps)/[MacOS](#macos_deps) dependencies**.

> Pip installation is not supported on Windows at the moment, but we hope some day it will.

To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
pip install vizdoom
```

To install the newest version from the repository:
```bash
pip install git+https://github.com/mwydmuch/ViZDoom.git
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
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON -DBUILD_JULIA=ON
make
```

where `-DBUILD_PYTHON=ON` and `-DBUILD_JULIA=ON` CMake options for Python and Julia bindings are optional (default OFF). To force building bindings for Python3 instead of the first version found use `-DBUILD_PYTHON3=ON`.

To build Julia binding you first need to install CxxWrap package by running `julia` and using `Pkg.add("CxxWrap")` command (see [Linux dependencies](#linux_deps)). Then you need to manually set `JlCxx_DIR` variable:

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
-DBUILD_JULIA=ON \
-DJlCxx_DIR=~/.julia/vX.X/CxxWrap/deps/usr/lib/cmake/JlCxx/
```


### <a name="macos_build"></a> MacOS

>>> Using [pip/conda](#pypi) is the recommended way to install ViZDoom, please try it first unless you are sure you want to compile the package by hand.

Run CMake and build generated Makefile.

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JULIA=ON
make
```

where `-DBUILD_PYTHON=ON` and `-DBUILD_JULIA=ON` CMake options for Python and Julia bindings are optional (default OFF). To force building bindings for Python3 instead of the first version found use `-DBUILD_PYTHON3=ON`.

Users with brew-installed Python/Anaconda **may** need to manually set `PYTHON_EXECUTABLE`, `PYTHON_INCLUDE_DIR`, `PYTHON_LIBRARY` variables:

It should look like this for brew-installed Python (use `-DBUILD_PYTHON3=ON`, `include/pythonX.Xm` and `lib/libpythonX.Xm.dylib` for Python 3):

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \
-DPYTHON_EXECUTABLE=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/bin/pythonX \
-DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/include/pythonX.X \
-DPYTHON_LIBRARY=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/libpythonX.X.dylib \
-DNUMPY_INCLUDES=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/pythonX.X/site-packages/numpy/core/include
```

Or for Anaconda (use `-DBUILD_PYTHON3=ON`, `include/pythonX.Xm` and `lib/libpythonX.Xm.dylib` for Python 3):

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \
-DPYTHON_EXECUTABLE=~/anacondaX/bin/pythonX \
-DPYTHON_INCLUDE_DIR=~/anacondaX/include/pythonX.X \
-DPYTHON_LIBRARY=~/anacondaX/lib/libpythonX.X.dylib \
-DNUMPY_INCLUDES=~/anacondaX/lib/pythonX.X/site-packages/numpy/core/include
```

To build Julia binding, you first need to install CxxWrap package by running `julia` and using `Pkg.add("CxxWrap")` command (see [MacOS dependencies](#macos_deps)). Then you need to manually set `JlCxx_DIR` variable:

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
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
* ZDoom dependencies paths

In configuration select BUILD_PYTHON, BUILD_PYTHON3 and BUILD_JAVA options for Python and Java bindings (optional, default OFF).

Use generated Visual Studio solution to build all parts of ViZDoom environment.


### <a name="output"></a> Compilation output
Compilation output will be placed in `build/bin` and it should contain following files.

* `bin/vizdoom / vizdoom.exe` - ViZDoom executable
* `bin/vizdoom.pk3` - resources file used by ViZDoom (needed by ViZDoom executable)
* `bin/libvizdoom.a / vizdoom.lib` - C++ ViZDoom static library
* `bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib` -  C++ ViZDoom dynamically linked library
* `bin/pythonX.X/vizdoom.so / vizdoom.pyd / vizdoom.dylib ` - ViZDoom Python X.X module
* `bin/pythonX.X/pip_package` - complete ViZDoom Python X.X package


### <a name="manual-install"></a> Manual installation

To manually install Python package copy `vizdoom_root_dir/build/bin/pythonX.X/pip_package` contents to `python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom`.
