FROM fedora:latest

WORKDIR vizdoom

# MINIMAL
RUN dnf update -y && dnf clean all && dnf install -y gcc gcc-c++ make cmake git boost-devel SDL2-devel openal-soft-devel python3-devel python3-pip

# FULL
#RUN dnf update -y && dnf clean all && dnf install -y gcc gcc-c++ make cmake boost-devel SDL2-devel freetype-devel openal-soft-devel python3-devel python3-pip

COPY . ./
CMD ["bash", "./scripts/build_and_test.sh"]
