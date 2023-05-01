# Installation and building

- [Installation and building](#installation-and-building)
  - [ Dependencies](#-dependencies)
    - [ Linux](#-linux)
    - [ MacOS](#-macos)
  - [ Building](#-building)
    - [ Windows](#-windows)
  - [ Installation via pip (recommended for Python users)](#-installation-via-pip-recommended-for-python-users)
  - [ Building manually (not recommended)](#-building-manually-not-recommended)
    - [ Linux / MacOS](#-linux--macos)
    - [ Windows](#-windows-1)
    - [ Compilation output](#-compilation-output)
    - [ Manual installation](#-manual-installation)


## <a name="deps"></a> Dependencies

Even if you plan to install ViZDoom via pip, you need to install some dependencies in your system first.


### <a name="linux_deps"></a> Linux
* CMake 3.4+
* Make
* GCC 6.0+
* Boost libraries 1.65.0+
* Python 3.7+ for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

To get all dependencies on apt-based Linux (Ubuntu, Debian, Linux Mint, etc.) execute the following commands in the shell (might require root access).
```bash
# All ZDoom dependencies (most are optional)
apt install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev

# Only essential ZDoom dependencies
apt install build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev

# Python 3 dependencies (alternatively Anaconda 3 installed)
apt install python3-dev python3-pip
# or install Anaconda 3 and add it to PATH
```

To get all dependencies on dnf/yum-based Linux (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.) execute the following commands in the shell (might require root access).
```bash
# Essential ZDoom dependencies
dnf install cmake git boost-devel SDL2-devel openal-soft-devel

# Python 3 dependencies (alternatively Anaconda 3 installed)
dnf install python3-devel python3-pip
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
* CMake 3.4+
* Clang 5.0+
* Boost libraries 1.65.0+
* Python 3.7+ for Python binding (optional)
## <a name="build"></a> Building

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get dependencies install [homebrew](https://brew.sh/)

```sh
# ZDoom dependencies and Boost libraries
brew install cmake boost openal-soft sdl2

# You can use system python or install Anaconda 3 and add it to PATH
```


### <a name="windows_deps"></a> Windows
* CMake 3.4+
* Visual Studio 2012+
* Boost 1.65+
* Python 3.7+ for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them (except Boost) are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).
You can download Boost from [here](https://www.boost.org/users/download).


## <a name="pypi"></a> Installation via pip (recommended for Python users)

ViZDoom for Python can be installed via **pip** on Linux, MacOS and Windows, and it is strongly recommended.
However you will still need to install **[Linux](#linux_deps)/[MacOS](#macos_deps) dependencies**, as it will be build locally from source.
For Windows 10 or 11 64-bit and Python 3.7+ we provide pre-build wheels (binary packages).


To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
pip install vizdoom
```

To install the newest version from the repository (only Linux and MacOS):
```bash
pip install git+https://github.com/mwydmuch/ViZDoom.git
```


## <a name="build"></a> Building manually (not recommended)

Instructions below can be used to build ViZDoom manually.

### <a name="linux_macos_build"></a> Linux / MacOS

>>> Using [pip](#pypi) is the recommended way to install ViZDoom, please try it first unless you are sure you want to compile the package by hand.

In ViZDoom's root directory:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_ENGINE=ON -DBUILD_PYTHON=ON
make
```

where `-DBUILD_ENGINE=ON` and `-DBUILD_PYTHON=ON` CMake options are optional (default ON).

### <a name="windows_build"></a> Windows

Run CMake GUI, select ViZDoom root directory and set paths to:
* BOOST_ROOT
* BOOST_INCLUDEDIR
* BOOST_LIBRARYDIR
* PYTHON_INCLUDE_DIR (optional, for Python/Anaconda bindings)
* PYTHON_LIBRARY (optional, for Python/Anaconda bindings)
* ZDoom dependencies paths

In configuration select `DBUILD_ENGINE` and `DBUILD_PYTHON` (optional, default ON).

Use generated Visual Studio solution to build all parts of ViZDoom environment.


### <a name="output"></a> Compilation output
Compilation output will be placed in `build/bin` and it should contain the following files.

* `bin/vizdoom / vizdoom.exe` - ViZDoom executable
* `bin/vizdoom.pk3` - resources file used by ViZDoom (needed by ViZDoom executable)
* `bin/libvizdoom.a / vizdoom.lib` - C++ ViZDoom static library
* `bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib` -  C++ ViZDoom dynamically linked library
* `bin/pythonX.X/vizdoom.so / vizdoom.pyd / vizdoom.dylib ` - ViZDoom Python X.X module
* `bin/pythonX.X/pip_package` - complete ViZDoom Python X.X package


### <a name="manual-install"></a> Manual installation

To manually install Python package copy `vizdoom_root_dir/build/bin/pythonX.X/pip_package` contents to `python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom`.
