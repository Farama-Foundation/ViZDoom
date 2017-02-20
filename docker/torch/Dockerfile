FROM ubuntu:16.04 # CPU ONLY

# GPU -> see: https://github.com/NVIDIA/nvidia-docker
#FROM nvidia/cuda:7.5-cudnn5-devel 
# or
#FROM nvidia/cuda:8.0-cudnn5-devel 

# Install all dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libsdl2-dev \
    libwildmidi-dev \
    nasm \
    nano \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev

# Clone Torch, install dependencies and build Torch (building blas may take a while)
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && \
    cd /root/torch && \
    bash install-deps && \
    ./install.sh
    
# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH
ENV TERM xterm

# Install more dependencies via luarocks
RUN luarocks install torchffi && \
    luarocks install image
    
# Install CUDA libraries
RUN luarocks install cutorch && \
    luarocks install cunn && \
    luarocks install cudnn

# Clone ViZDoom and build + install via luarocks
RUN git clone https://github.com/mwydmuch/ViZDoom.git /root/vizdoom && \
    cd /root/vizdoom && \
    luarocks make

# Expose ViZDoom's multiplayer port
EXPOSE 5029
 
# Code below allows sharing X11 socket with container
# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer
ENV HOME /home/developer

# Set workdir
WORKDIR /root/vizdoom/examples/lua

# Expose th command
CMD ["th"]

# Examples of usage:
# BUILD THIS DOCKER: docker build -t torch+vizdoom .
# RUN THIS DOCKER WITH X11: docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix torch+vizdoom th basic.lua

