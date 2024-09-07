# Python quick start


## Linux
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
Both x86-64 and AArch64 (ARM64) architectures are supported.
Wheels are available for Python 3.8+ on Linux.

### Audio buffer requirement
If you want to use audio buffer, you need to have OpenAL library installed.
It is installed by default in many desktop distros. Otherwise it can be installed from the package manager.
On apt-based distros (Ubuntu, Debian, Linux Mint, etc.), you can install it by running:
```sh
apt install libopenal1
```
And on dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.), you can install it by running:
```sh
dnf install openal-soft
```
On RHEL/CentOS/Alma/Rocky Linux 9, you may need first enable crb repository by running `dnf --enablerepo=crb install`.

### Installing from source distribution on Linux
If Python wheel is not available for your platform (distros incompatible with manylinux_2_28 standard), pip will try to install (build) ViZDoom from the source.
ViZDoom requires a C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional), and Python 3.8+ to install from source.
Below, you will find instructions on how to install these dependencies.

#### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)
To build ViZDoom run (it may take a few minutes):
```sh
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev
pip install vizdoom
```
We recommend using at least Ubuntu 18.04+ or Debian 10+ with Python 3.7+.

#### dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)
To install ViZDoom, run (it may take a few minutes):
```sh
dnf install cmake git boost-devel SDL2-devel openal-soft-devel
pip install vizdoom
```
We recommend using at least Fedora 35+ or RHEL/CentOS/Alma/Rocky Linux 9+ with Python 3.8+.
To install openal-soft-devel on RHEL/CentOS/Alma/Rocky Linux 9, one needs to enable crb repository first by running `dnf --enablerepo=crb install`.

### Installing master branch version
To install the master branch version of ViZDoom run:
```sh
pip install git+https://github.com/Farama-Foundation/ViZDoom
```
It requires the to have the above dependencies installed.


## macOS
To install the latest release of ViZDoom, just run (it may take a few minutes as it will build ViZDoom from source on M1/M2 chips):
```sh
pip install vizdoom
```
Both Intel and Apple Silicon CPUs are supported.
Pre-build wheels are available for Intel macOS 12.0+ and Apple Silicon macOS 14.0+.

If Python wheel is not available for your platform (Python version <3.8, older macOS version), pip will try to install (build) ViZDoom from the source.
In this case, install the required dependencies using Homebrew:
```sh
brew install cmake boost sdl2 openal-soft
```
We recommend using at least macOS High Sierra 10.13+ with Python 3.8+.
On Apple Silicon (M1, M2, and M3), make sure you are using Python/Pip for Apple Silicon.

To install the master branch version of ViZDoom, run, in this case you also need to have the above dependencies installed:
```sh
pip install git+https://github.com/Farama-Foundation/ViZDoom
```


## Windows
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
At the moment, only x86-64 architecture is supported on Windows.
Wheels are available for Python 3.9+ x86-64 on Windows.

Please note that the Windows version is not as well-tested as Linux and macOS versions.
It can be used for development and testing but if you want to conduct serious (time and resource-extensive) experiments on Windows,
please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl) with Linux version.
Windows version is bundled with OpenAL library, so you don't need to install it separately.
