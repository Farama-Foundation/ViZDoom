FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

WORKDIR vizdoom

# MINIMAL
RUN apt update && apt install -y build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev python3-dev python3-pip

# FULL
#RUN apt update && apt install -y build-essential cmake libboost-all-dev libsdl2-dev libfreetype-dev libopenal-dev python3-dev python3-pip

COPY . ./
CMD ["bash", "./scripts/build_and_test.sh"]
