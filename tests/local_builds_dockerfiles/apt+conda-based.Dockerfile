FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

WORKDIR /vizdoom

# Install wget
RUN apt-get update && apt-get install -y build-essential git make cmake wget

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

# Install conda dependencies
RUN conda install -y -c conda-forge boost sdl2 openal-soft

COPY . ./
CMD ["bash", "./scripts/build_and_test_conda.sh"]
