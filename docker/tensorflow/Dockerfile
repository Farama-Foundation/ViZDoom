FROM ubuntu:16.04

FROM nvidia/cuda:8.0-cudnn5-devel

# ViZdoom dependencies
RUN apt-get update
RUN apt-get install -y \
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
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev

# Python3
RUN apt-get install -y python3-dev python3 python3-pip
RUN pip3 install pip --upgrade


# Enables X11 sharing and creates user home directory
ENV USER_NAME vizdoom
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}


RUN git clone https://github.com/mwydmuch/ViZDoom ${HOME_DIR}/vizdoom
RUN pip3 install Cython
RUN pip3 install ${HOME_DIR}/vizdoom
RUN pip3 install tensorflow-gpu
RUN pip3 install matplotlib scipy scikit-image tqdm

USER ${USER_NAME}
WORKDIR ${HOME_DIR}


CMD cd vizdoom/examples/python; python3 learning_tensorflow.py
