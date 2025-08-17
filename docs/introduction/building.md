# Building from source

Here we describe how to build ViZDoom from source.
If you want to install pre-build ViZDoom wheels for Python, see [Python quick start](./python_quickstart.md).


## Dependencies

To build ViZDoom (regardless of the method), you need to install some dependencies in your system first.


### Linux

To build ViZDoom on Linux, the following dependencies are required:
* CMake 3.12+
* Make
* GCC 6.0+
* Boost libraries 1.54.0+
* Python 3.7+ for Python binding (optional)

Also some of additionally [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

#### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

To get all dependencies on apt-based Linux (Ubuntu, Debian, Linux Mint, etc.) execute the following commands in the shell (might require root access).
```sh
# All possible ViZDoom dependencies,
# most are optional and required only to support alternative sound and music backends in the engine
# other can replace libraries that are included in the ViZDoom repository
apt install build-essential cmake git libsdl2-dev libboost-all-dev libopenal-dev \
zlib1g-dev libjpeg-dev tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev \
timidity libwildmidi-dev unzip

# Only essential ViZDoom dependencies
apt install build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev

# Python 3 dependencies (alternatively Anaconda 3 installed)
apt install python3-dev python3-pip
# or install Anaconda 3 and add it to PATH
```

#### dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)

To get all dependencies on dnf/yum-based Linux (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.) execute the following commands in the shell (might require root access).
```sh
# Essential ZDoom dependencies
dnf install cmake git boost-devel SDL2-devel openal-soft-devel

# Python 3 dependencies (alternatively Anaconda 3 installed)
dnf install python3-devel python3-pip
```

#### Anaconda/Miniconda

If you do not have a root access, you can use a conda (e.g. [miniconda](https://docs.conda.io/en/latest/miniconda.html)) environment to install dependencies to your environment only:
```sh
conda install -c conda-forge boost cmake gtk2 sdl2 openal-soft
```

Note that to install ViZDoom in a conda environment you have to pull, build and install ViZDoom manually with
```
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
python setup.py build && python setup.py install
```


### MacOS
To build ViZDoom on MacOS, the following dependencies are required:
* CMake 3.12+
* Clang 5.0+
* Boost libraries 1.54.0+
* Python 3.7+ for Python binding (optional)

Also some of additionally [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get all the dependencies install [homebrew](https://brew.sh/) first, than execute the following commands in the shell:
```sh
brew install cmake boost sdl2 openal-soft
```

⚠️ On Apple Silicon (M-series chip), make sure you are using homebrew for `arm64`.

Here is a helpful command to set up the environment variables for homebrew on Apple Silicon if both `arm64` and `x86` versions were installed:
```sh
eval "$(/opt/homebrew/bin/brew shellenv)"
```

We recommend using at least macOS High Sierra 10.13+ with Python 3.8+.
On Apple Silicon, make sure you are using Python/Pip for Apple Silicon.


### Windows
* CMake 3.12+
* Visual Studio 2012+
* Boost libraries 1.54.0+
* Python 3.7+ for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them (except Boost) are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).
You can download Boost from [here](https://www.boost.org/users/download).



## Building via pip (recommended for Python users)

ViZDoom for Python can be build via **pip** on Linux, MacOS and Windows, and it is strongly recommended.
Even when building using pip you still need to install dependencies.

To build the newest version from the repository run:
```sh
pip install git+https://github.com/mwydmuch/ViZDoom.git
```
or
```sh
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
pip install .
```

On Linux and MacOS dependencies should be found automatically.
On Windows you need to manually set following environment variables:
* `BOOST_ROOT` - the path to the directory with Boost libraries (e.g. `C:\boost_1_76_0`),
* `VIZDOOM_BUILD_GENERATOR_NAME` - generator name (e.g. `Visual Studio 16 2019`),
* `VIZDOOM_WIN_DEPS_ROOT` - the path to the directory with ZDoom dependencies (e.g. `C:\ViZDoomWinDepBin`).

The process of building ViZDoom this way on Windows is demonstarted in [scripts/windows_build_wheels.bat](https://github.com/Farama-Foundation/ViZDoom/tree/master/scripts/windows_build_wheels.bat).


## Building Python Type Stubs (`vizdoom.pyi`)

To enable Python typing support, the creation of a Python Interface file `vizdoom.pyi` is now part of the build process.

To ensure `vizdoom.pyi` is properly created, the following dependencies are required:
* pybind11-stubgen
* black (optional)
* isort (optional)

They can all be installed via pip:
```sh
pip install pybind11-stubgen black isort
```
To test the generated stub manually:
```
stubtest vizdoom --allowlist stubtest_allowlists.txt
```


## Building manylinux wheels

To build manylinux wheels you need to install docker and cibuildwheel. Then on Linux and MacOS run in ViZDoom root directory:
```sh
cibuildwheel --platform linux
```

The binary ViZDoom wheels will be placed in `wheelhouse` directory.
In case of building using cibuildwheel, the dependencies are installed automatically inside the docker container, so you do not need to install them manually in your system.


## Building manually (not recommended)

Instructions below can be used to build ViZDoom manually.
We recommend doing it only if you want to use C++ API, work on the ViZDoom, or if you have problems with pip installation.

### Linux / MacOS
In ViZDoom's root directory:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_ENGINE=ON -DBUILD_PYTHON=ON -DCREATE_PYTHON_STUBS=OFF
make
```

where `-DBUILD_ENGINE=ON` and `-DBUILD_PYTHON=ON`  CMake options are optional (default ON). Setting `-DCREATE_PYTHON_STUBS=OFF` will skip the creation of `vizdoom.pyi` (which is OFF by default).


### Windows
1. Run CMake GUI or cmake command in cmd/powershell in ViZDoom root directory with the following paths provided:
* BOOST_ROOT
* BOOST_INCLUDEDIR
* BOOST_LIBRARYDIR
* PYTHON_INCLUDE_DIR (optional, for Python/Anaconda bindings)
* PYTHON_LIBRARY (optional, for Python/Anaconda bindings)
* ZDoom dependencies paths

1. In configuration select `DBUILD_ENGINE`, `DBUILD_PYTHON` (optional, default ON) and `DCREATE_PYTHON_STUBS` (optional, default OFF).

2. Use generated Visual Studio solution to build all parts of ViZDoom environment.

The process of building ViZDoom this way on Windows is demonstarted in [scripts/windows_build_cmake.bat](https://github.com/Farama-Foundation/ViZDoom/tree/master/scripts/windows_build_cmake.bat) script.


### Compilation output
Compilation output will be placed in `build/bin` and it should contain the following files.

* `bin/vizdoom / vizdoom.exe` - ViZDoom executable
* `bin/vizdoom.pk3` - resources file used by ViZDoom (needed by ViZDoom executable)
* `bin/libvizdoom.a / vizdoom.lib` - C++ ViZDoom static library
* `bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib` -  C++ ViZDoom dynamically linked library
* `bin/pythonX.X/vizdoom.so / vizdoom.pyd / vizdoom.dylib ` - ViZDoom Python X.X module
* `bin/pythonX.X/vizdoom` - complete ViZDoom Python X.X package


### Manual installation
To manually install Python package copy `vizdoom_root_dir/build/bin/pythonX.X/vizdoom` contents to `python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom`.
