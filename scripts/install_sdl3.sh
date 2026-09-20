#!/usr/bin/env bash
# Build SDL3 on Linux distributions without an SDL3 development package.
set -euo pipefail

if command -v apt-get >/dev/null; then
    apt-get update
    apt-get install -y build-essential cmake curl pkg-config \
        libasound2-dev libpulse-dev libx11-dev libxext-dev libxrandr-dev \
        libxcursor-dev libxfixes-dev libxi-dev libxss-dev libxtst-dev \
        libgl1-mesa-dev libegl1-mesa-dev libudev-dev libdbus-1-dev \
        libwayland-dev libxkbcommon-dev
else
    yum install -y cmake curl gcc gcc-c++ make pkgconf-pkg-config \
        alsa-lib-devel pulseaudio-libs-devel libX11-devel libXext-devel \
        libXrandr-devel libXcursor-devel libXfixes-devel libXi-devel \
        libXScrnSaver-devel libXtst-devel mesa-libGL-devel mesa-libEGL-devel \
        systemd-devel dbus-devel wayland-devel libxkbcommon-devel
fi

sdl_version=3.2.30
sdl_build_dir=$(mktemp -d)
trap 'rm -rf "$sdl_build_dir"' EXIT
curl -fL "https://www.libsdl.org/release/SDL3-${sdl_version}.tar.gz" \
    -o "$sdl_build_dir/SDL3.tar.gz"
echo "4c3b09330d866dc52eb65b66259a6684ad387252ca8c57901b3a2b534eb42e3d  $sdl_build_dir/SDL3.tar.gz" | sha256sum -c -
tar -xzf "$sdl_build_dir/SDL3.tar.gz" -C "$sdl_build_dir"
cmake -S "$sdl_build_dir/SDL3-${sdl_version}" -B "$sdl_build_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib \
    -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TEST_LIBRARY=OFF
cmake --build "$sdl_build_dir/build" --parallel 4
cmake --install "$sdl_build_dir/build"
ldconfig
