# Python quick start

## Linux
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
Both x86-64 and AArch64 (ARM64) architectures are supported.

If Python wheel is not available for your platform (Python version <3.8, distros below manylinux_2_28 standard), pip will try to install (build) ViZDoom from source.
ViZDoom requires C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional) and Python 3.7+. Below you will find instructrion how to install these dependencies.

### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

To build ViZDoom run (it may take few minutes):
```sh
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev
pip install vizdoom
```
We recommend using at least Ubuntu 18.04+ or Debian 10+ with Python 3.7+.

### dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)

To install ViZDoom run (it may take few minutes):
```sh
dnf install cmake git boost-devel SDL2-devel openal-soft-devel
pip install vizdoom
```
We recommend using at least Fedora 35+ or RHEL/CentOS/Alma/Rocky Linux 9+ with Python 3.7+.
To install openal-soft-devel on RHEL/CentOS/Alma/Rocky Linux 9, one needs to use `dnf --enablerepo=crb install`.


## macOS
To install the latest release of ViZDoom just run (may take few minutes as it will build ViZDoom from source on M1/M2 chips):
```sh
brew install cmake boost sdl2 openal-soft
pip install vizdoom
```
Both Intel and Apple Silicon CPUs are supported.
We recommend using at least macOS High Sierra 10.13+ with Python 3.7+.
On Apple Silicon (M1 and M2), make sure you are using Python/Pip for Apple Silicon.


## Windows
To install the latest release of ViZDoom, just run:
```sh
pip install vizdoom
```
At the moment only x86-64 architecture is supported on Windows.

Please note that the Windows version is not as well-tested as Linux and macOS versions.
It can be used for development and testing but if you want to conduct serious (time and resource-extensive) experiments on Windows,
please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with Linux version.
