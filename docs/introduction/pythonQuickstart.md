# Python quick start

## Linux
Both x86-64 and ARM64 architectures are supported.
ViZDoom requires C++11 compiler, CMake 3.12+, Boost 1.65+ SDL2, OpenAL (optional) and Python 3.8+. Below you will find instructrion how to install these dependencies.

### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

To install ViZDoom run (may take few minutes):
```
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev
pip install vizdoom
```
We recommend using at least Ubuntu 18.04+ or Debian 10+ with Python 3.8+.

### dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)

To install ViZDoom run (may take few minutes):
```
dnf install cmake git boost-devel SDL2-devel openal-soft-devel
pip install vizdoom
```
We recommend using at least Fedora 35+ or RHEL/CentOS/Alma/Rocky Linux 9+ with Python 3.8+. To install openal-soft-devel on RHEL/CentOS/Alma/Rocky Linux 9, one needs to use `dnf --enablerepo=crb install`.

### Conda-based installation
To install ViZDoom on a conda environment (no system-wide installations required):
```
conda install -c conda-forge boost cmake sdl2
git clone https://github.com/mwydmuch/ViZDoom.git --recurse-submodules
cd ViZDoom
python setup.py build && python setup.py install
```
Note that `pip install vizdoom` won't work with conda install and you have to follow these steps.


## macOS
Both Intel and Apple Silicon CPUs are supported.

To install ViZDoom on run (may take few minutes):
```
brew install cmake git boost openal-soft sdl2
pip install vizdoom
```
We recommend using at least macOS High Sierra 10.13+ with Python 3.8+.
On Apple Silicon (M1 and M2), make sure you are using Python for Apple Silicon.


## Windows
To install pre-build release for Windows 10 or 11 64-bit and Python 3.8+ just run (should take few seconds):
```
pip install vizdoom
```

Please note that the Windows version is not as well-tested as Linux and macOS versions. It can be used for development and testing if you want to conduct experiments on Windows, please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
