FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

WORKDIR v/izdoom

# Install minimal dependencies
RUN apt update && apt install -y build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev python3-dev python3-pip
#RUN apt update && apt install -y build-essential cmake libboost-all-dev libsdl2-dev libfreetype-dev libopenal-dev python3-dev python3-pip

# Install all possible dependencies
#RUN apt update && apt install -y build-essential cmake git libsdl2-dev libboost-all-dev libopenal-dev zlib1g-dev libjpeg-dev tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev timidity libwildmidi-dev unzip

COPY . ./
CMD ["bash", "./scripts/build_and_test.sh"]
