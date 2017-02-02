# Installation/Building

* [Installation via PyPI(pip)](#pypi)
* [Installation via LuaRocks](#luarocks)
* [Building on Linux](#linux)
* [Building on Windows](#windows)
* [Building on MacOS/OSX](#osx)
* [Compilation output](#output)


## <a name="pypi"></a> Installation via PyPI (recommended for Python users)

ViZDoom for Python can be installed via **pip** on Linux and MacOS and it is strongly recommended. However you will still need to install the  **[dependencies](#linux_deps)**. Without pip installation you need to have vizdoom.so in the execution directory and specify paths to vizdoom and freedoom2.wad manually which is quite annoying.

> Pip installation is not supported on Windows at the moment but soon it will.

To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
# use pip3 for python3
sudo pip install vizdoom
```
To install newest version from the repository:
```bash
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
# use pip3 for python3
sudo pip install .
```
Or without cloning yourself:
```bash
# use pip3 for python3
sudo pip install git+https://github.com/mwydmuch/ViZDoom
```

## <a name="luarocks"></a> Installation via LuaRocks (recommended for Torch7 users)

ViZDoom for Python can be installed via **luarocks** on Linux and MacOS and it is strongly recommended. However you will still need to install the  **[dependencies](#linux_deps)**.

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


## <a name="linux"></a> Linux

### <a name="linux_deps"></a>Dependencies
* CMake 2.8+
* Make
* GCC 4.6+
* Boost libraries (tested on 1.54, 1.58, 1.59, 1.61)
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)
* LUA 5.1 for Lua binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

To get all dependencies (except JDK) on Ubuntu execute the following commands in the shell (requires root access):
```bash
# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev 

# Python 2 and Python 3 bindings dependencies
sudo apt-get install python-pip python3-pip 
pip install numpy #just for python2 binding
pip3 install numpy #just for python3 binding

# Boost libraries
sudo apt-get install libboost-all-dev

# Lua binding dependencies
sudo apt-get install liblua5.1-dev
```

### Compiling
In ViZDoom's root directory:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_JAVA=ON -DBUILD_LUA=ON
make
```

``-DBUILD_PYTHON=ON`` and ``-DBUILD_JAVA=ON`` and ``-DBUILD_LUA=ON`` CMake options for Python, Java and Lua bindings are optional (default OFF). To force building bindings for Python3 instead of first version found use ``-DBUILD_PYTHON3=ON`` (needs Boost.Python builded with Python 3, default OFF).


## <a name="windows"></a> Windows

> Setting up the compilation on Windows is really tedious so using the precompiled binaries is recommended.

### Dependencies
* CMake 2.8+
* Visual Studio 2012+
* Boost libraries
* Python 2.7+ or Python 3.4+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.

### Compiling
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


## <a name="osx"></a>OSX

### Dependencies
* CMake 2.8+
* XCode 5+
* Boost libraries
* Python 2.7+ or Python 3+ with Numpy and Boost.Python for Python binding (optional)
* JDK for Java binding (JAVA_HOME must be set) (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

### Compiling
Run CMake and use generated project.

Users with brew-installed Python may need to manually set:
``-DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/x.x.x/Frameworks/Python.framework/Versions/x.x/include/pythonx.x`` and 
``-DPYTHON_LIBRARY=/usr/local/Cellar/python/x.x.x/Frameworks/Python.framework/Versions/x.x/lib/libpythonx.x.dylib``


## <a name="output"></a> Compilation output
Compilation output will be placed in ``vizdoom_root_dir/bin`` and it should contain following files.

* ``bin/vizdoom / vizdoom.exe`` - ViZDoom executable
* ``bin/vizdoom.pk3`` - resources file used by ViZDoom (needed by ViZDoom executable)
* ``bin/libvizdoom.a / vizdoom.lib`` - C++ ViZDoom static library
* ``bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib`` -  C++ ViZDoom dynamically linked library
* ``bin/python2/vizdoom.so / vizdoom.pyd / vizdoom.dylib`` - ViZDoom Python module
* ``bin/python3/vizdoom.so / vizdoom.pyd / vizdoom.dylib`` - ViZDoom Python3 module
* ``bin/lua/vizdoom.so / vizdoom.so / vizdoom.dylib`` - ViZDoom Lua C module
* ``bin/java/libvizdoom.so / vizdoom.dll / libvizdoom.dylib`` -  ViZDoom library for Java
* ``bin/java/vizdoom.jar`` -  Contains ViZDoom Java classes
* ``bin/lua/vizdoom`` - ViZDoom Lua module


## Torch7 Lua bindings (Linux and MacOS)
If you want to build against LuaJIT installed locally by Torch (as in http://torch.ch/docs/getting-started.html#_):
```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_LUA=ON -DLUA_LIBRARIES=torch_root_dir/install/lib/libluajit.so/dylib -DLUA_INCLUDE_DIR=torch_root_dir/install/include/
```

Manual installation: 
Copy `vizdoom_root_dir/bin/lua/luarocks_package` contents to `torch_root_dir/install/lib/lua/5.1/vizdoom` 
and `vizdoom_root_dir/bin/lua/luarocks_shared_package` contents to `torch_root_dir/install/share/lua/5.1/vizdoom`.
