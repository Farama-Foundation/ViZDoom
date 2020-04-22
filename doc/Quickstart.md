# Quickstart

> This guide can be a little bit outdated.

* [Quickstart for MacOS with Anaconda3](#quickstart_macos_anaconda)

TODO:
* Quickstart for Ubuntu 
* Quickstart for Windows
* Change "Quickstart for MacOS" to use PyPI

## <a name="quickstart_macos_anaconda"></a> Quickstart for MacOS and Anaconda3 (Python 3.6)

Install [homebrew](https://brew.sh/), and [Anaconda3](https://www.continuum.io/downloads) (Python 3.6).

1. Install dependencies

```sh
brew install \
    cmake \
    boost \
    boost-python --with-python3
```

2. Clone the repository

```sh
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom
```

3. Run cmake against anaconda and native boost libraries. 

The library version numbers will change over time (e.g. `python3.6m`, `boost-python/1.64.0`) and may need to be updated. 

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON3=ON \
-DPYTHON_INCLUDE_DIR=$HOME/anaconda3/include/python3.6m \ 
-DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.6m.dylib \
-DPYTHON_EXECUTABLE=$HOME/anaconda3/bin/python3 \
-DBOOST_PYTHON3_LIBRARY=/usr/local/Cellar/boost-python/1.64.0/lib/libboost_python3.dylib \
-DNUMPY_INCLUDES=$HOME/anaconda3/lib/python3.6/site-packages/numpy/core/include
```

4. Build it!

```sh
make
```

5. Move the output to anaconda's `site-packages` directory, or your local virtual environments `site-packages` directory. 

```sh
mv -r build/bin/python3/pip_package/ $HOME/anaconda3/lib/python3.6/site-packages/vizdoom
```

6. Test if it works.

```sh
cd examples/python
./basic.py
```
