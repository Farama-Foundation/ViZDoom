FROM ubuntu:16.04 # CPU ONLY

# GPU -> see: https://github.com/NVIDIA/nvidia-docker
#FROM nvidia/cuda:7.5-cudnn5-devel 
# or
#FROM nvidia/cuda:8.0-cudnn5-devel 

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng12-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    python-dev \
    python \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev


# Pick up pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install more dependencies via pip
RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \

# Install TensorFlow
RUN pip --no-cache-dir install http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
# or
#RUN pip --no-cache-dir install http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

# Clone ViZDoom and build + install via pip
RUN git clone https://github.com/mwydmuch/ViZDoom.git /root/vizdoom && \
    pip install git+file:///root/vizdoom

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
WORKDIR /root/vizdoom/examples/pyhton

# Expose python command
CMD ["python"]

# Examples of usage:
# BUILD THIS DOCKER: docker build -t tf+vizdoom .
# RUN THIS DOCKER WITH X11: docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix tf+vizdoom python basic.py

