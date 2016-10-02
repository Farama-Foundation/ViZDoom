# Building ViZDoom for python3

## Python interpreter
Make sure that you have python3 intepreter with Numpy installed
````bash
$ apt-get install python3 python3-pip
$ pip3 install numpy
```
## Boost
libboost-python comes by deafult with python2 support. To change this you need to recompile Boost from sources. First, download sources of (preferably) the newest Boost library from [somewhere here](http://www.boost.org/users/download/). Enter the downloaded directory and run:

```bash
$ ./bootstrap.sh --with-python=python3.X
```
3.X means  your actual version e.g. 3.5 or 3.4.

Then run:
```bash
$ ./b2
$ sudo ./b2 install
```
Now you should be able to build Boost bindings for python3.

## ViZDoom Cmake
To build python3 bindings set BUILD_PYTHON and BUILD_PYTHON3 flags to true. Since building against 2 versions at the some time is currently not supported it will result in building python3 binding **instead of** python2. Resulting binaries will be placed in **bin/python3** and appropriate link will be made for python examples. 

## python2 and python3 at the same time

TODO Coming soon

