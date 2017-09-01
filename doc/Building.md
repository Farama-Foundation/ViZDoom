# Installation/Building

* [Dependencies](#deps)
  * [Linux](#linux_deps)
  * [MacOS/OSX](#osx_deps)
  * [Windows](#windows_deps)
* [Installation via PyPI(pip/conda)](#pypi)
* [Installation via LuaRocks](#luarocks)
* [Building](#build)
  * [Linux](#linux_build)
  * [MacOS/OSX](#osx_build)
  * [Windows](#windows_build)
  * [Compilation output](#output)
  * [Building against Anaconda](#anaconda)
  * [Building against Torch's LuaJIT](#torchluajit)


## <a name="deps"></a> Dependencies

Even if you plan to install ViZDoom via PyPI or LuaRocks you need to install some dependencies in your system first.


### <a name="linux_deps"></a> Linux
* CMake 2.8+
* Make
* GCC 4.6+
* Boost libraries 1.54+
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.


To get all dependencies (except JDK) on Ubuntu execute the following commands in the shell (requires root access):
```bash
# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev 

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 2 dependencies
sudo apt-get install python-pip
pip install numpy
# or install Anaconda 2 and add it to PATH

# Python 3 dependencies
sudo apt-get install python3-pip 
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Lua binding dependencies
sudo apt-get install liblua5.1-dev 
# Lua shipped with Torch can be used instead, so it isn't needed if installing via LuaRocks
```

To get Torch see: [Getting started with Torch](http://torch.ch/docs/getting-started.html).


### <a name="osx_deps"></a> MacOS/OSX
* CMake 2.8+
* XCode 5+
* Boost libraries
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get dependencies install [homebrew](https://brew.sh/)

```sh
# ZDoom dependencies and Boost libraries
brew install cmake boost sdl2

# Python 2 dependecies
brew install boost-python
pip install numpy

# Python 3 dependecies
brew install boost-python --with-python3
pip3 install numpy
```

To get Torch see: [Getting started with Torch](http://torch.ch/docs/getting-started.html).


### <a name="windows_deps"></a> Windows
* CMake 2.8+
* Visual Studio 2012+
* Boost libraries
* Python 2.7+ or Python 3.4+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* Torch7 or LUA 5.1 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).


## <a name="pypi"></a> Installation via PyPI (recommended for Python users)

ViZDoom for Python can be installed via **pip/conda** on Linux and MacOS and it is strongly recommended. 
However you will still need to install **[Linux](#linux_deps)/[MacOS](#osx_deps) dependencies**. 

> Pip installation is not supported on Windows at the moment but we hope some day it will.

To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
# use pip3 for Python3 or conda for Anaconda version added to PATH
sudo pip install vizdoom
```

To install newest version from the repository:
```bash
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
# use pip3 for Python3 or conda for Anaconda version added to PATH
sudo pip install .
```

Or without cloning yourself:
```bash
# use pip3 for Python3 or conda for Anaconda version added to PATH
sudo pip install git+https://github.com/mwydmuch/ViZDoom
```

If you have a problem with missing Boost.Python **(new Anaconda with Python 3.6)** instead of building it on your own, clone local_boost_python branch to install ViZDoom with it's own version of library
```bash
git clone https://github.com/mwydmuch/ViZDoom -b local_boost_python
cd ViZDoom
# use pip3 for Python3 or conda for Anaconda version added to PATH
sudo pip install .
```


## <a name="luarocks"></a> Installation via LuaRocks (recommended for Torch7 users)

ViZDoom for Torch7 can be installed via **luarocks** on Linux and MacOS and it is strongly recommended. 
However you will still need to install **[Linux](#linux_deps)/[MacOS](#osx_deps) dependencies**. 

To install the most stable official release from [LuaRocks](https://pypi.python.org/pypi):
```bash
luarocks install vizdoom
```

To install newest version from the repository:
```bash
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
luarocks make
```


## <a name="build"></a> Building

### <a name="linux"></a> Linux

In ViZDoom's root directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON
make
```

``-DBUILD_PYTHON=ON`` and ``-DBUILD_JAVA=ON`` and ``-DBUILD_LUA=ON`` CMake options for Python, Java and Lua bindings are optional (default OFF). To force building bindings for Python3 instead of first version found use ``-DBUILD_PYTHON3=ON`` (needs Boost.Python builded with Python 3, default OFF).


### <a name="osx"></a>OSX

Run CMake and use generated project.

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON
make
```

``-DBUILD_PYTHON=ON`` and ``-DBUILD_JAVA=ON`` and ``-DBUILD_LUA=ON`` CMake options for Python, Java and Lua bindings are optional (default OFF). To force building bindings for Python3 instead of first version found use ``-DBUILD_PYTHON3=ON`` (needs Boost.Python builded with Python 3, default OFF).

Users with brew-installed Python/Anaconda may need to manually set PYTHON_EXECUTABLE, PYTHON_INCLUDE_DIR, PYTHON_LIBRARY:

It should look like this for brew-installed Python:

```
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \ (or -DBUILD_PYTHON3=ON for Python 3)
-DPYTHON_EXECUTABLE=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/bin/pythonx
-DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/include/pythonx.x
-DPYTHON_LIBRARY=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/libpythonx.x.dylib
-DNUMPY_INCLUDES=/usr/local/Cellar/python/X.X.X/Frameworks/Python.framework/Versions/X.X/lib/pythonX.X/site-packages/numpy/core/include
```

Or for Anaconda:

```
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \ (or -DBUILD_PYTHON3=ON for Python 3)
-DPYTHON_INCLUDE_DIR=$HOME/anacondaX/include/pythonX \ 
-DPYTHON_LIBRARY=$HOME/anacondaX/lib/libpythonX.dylib \
-DPYTHON_EXECUTABLE=$HOME/anacondaX/bin/pythonX \
-DNUMPY_INCLUDES=$HOME/anacondaX/lib/pythonX.X/site-packages/numpy/core/include
```


### <a name="windows_build"></a> Windows

Setting up the compilation on Windows is really tedious so using the precompiled binaries is recommended.

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
Compilation output will be placed in ``vizdoom_root_dir/bin`` and it should contain following files.

* ``bin/vizdoom / vizdoom.exe`` - ViZDoom executable
* ``bin/vizdoom.pk3`` - resources file used by ViZDoom (needed by ViZDoom executable)
* ``bin/libvizdoom.a / vizdoom.lib`` - C++ ViZDoom static library
* ``bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib`` -  C++ ViZDoom dynamically linked library
* ``bin/python2/vizdoom.so / vizdoom.pyd / vizdoom.dylib`` - ViZDoom Python 2 module
* ``bin/python3/vizdoom.so / vizdoom.pyd / vizdoom.dylib`` - ViZDoom Python 3 module
* ``bin/python2/pip_package`` - complete ViZDoom Python 2 package
* ``bin/python3/pip_package`` - complete ViZDoom Python 3 package
* ``bin/lua/vizdoom.so / vizdoom.so / vizdoom.dylib`` - ViZDoom Lua C module
* ``bin/lua/luarocks_package`` - complete ViZDoom Torch package
* ``bin/java/libvizdoom.so / vizdoom.dll / libvizdoom.dylib`` -  ViZDoom library for Java
* ``bin/java/vizdoom.jar`` -  Contains ViZDoom Java classes


### <a name="manual-install"></a> Manual installation

To manually install Python package copy `vizdoom_root_dir/bin/pythonX/pip_package` contents to `python_root_dir/lib/pythonx.x/site-packages/site-packages/vizdoom`.

To manually install Torch package copy `vizdoom_root_dir/bin/lua/luarocks_package` contents to `torch_root_dir/install/lib/lua/5.1/vizdoom` and `vizdoom_root_dir/bin/lua/luarocks_shared_package` contents to `torch_root_dir/install/share/lua/5.1/vizdoom`.


### <a name="anaconda"></a> Building Anaconda bindings (Linux and MacOS)

> If you don't have any special reason then use [PyPI way](#pypi).

If you want to build against Anaconda, you may need to manually set:

```
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON=ON \ (or -DBUILD_PYTHON3=ON for Python 3)
-DPYTHON_INCLUDE_DIR=/path_to_anaconda_root_dir/include/pythonx.x \
-DPYTHON_LIBRARY=/path_to_anaconda_root_dir/lib/libpythonx.x.so (or dll/dylib) \
-DPYTHON_EXECUTABLE=/path_to_anaconda_root_dir/bin/pythonx \
-DBOOST_PYTHON_LIBRARY=/path_to_boost_lib_dir/libboost_python-pyxx.so (or dll/dylib) \ (or -DBOOST_PYTHON3_LIBRARY=... for Anaconda with Python 3)
-DNUMPY_INCLUDES=/path_to_anaconda_root_dir/lib/pythonx.x/site-packages/numpy/core/include
```

Manual installation:
Copy `vizdoom_root_dir/bin/python2/pip_package` or `vizdoom_root_dir/bin/python3/pip_package` to `anaconda_root_dir/lib/pythonx.x/site-packages/vizdoom`.


### <a name="torchluajit"></a> Building Torch7 Lua bindings (Linux and MacOS)

> If you don't have any special reason then use [LuaRocks way](#luarocks).

If you want to build against LuaJIT installed locally by Torch (as in http://torch.ch/docs/getting-started.html#_):
```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_LUA=ON -DLUA_LIBRARIES=torch_root_dir/install/lib/libluajit.so/dylib -DLUA_INCLUDE_DIR=torch_root_dir/install/include/
```

Manual installation: 
Copy `vizdoom_root_dir/bin/lua/luarocks_package` contents to `torch_root_dir/install/lib/lua/5.1/vizdoom` and `vizdoom_root_dir/bin/lua/luarocks_shared_package` contents to `torch_root_dir/install/share/lua/5.1/vizdoom`.
