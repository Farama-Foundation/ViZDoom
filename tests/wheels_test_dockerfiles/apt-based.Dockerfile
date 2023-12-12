FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

WORKDIR /vizdoom

# Install Python and pip
RUN apt update && apt install -y python3-dev python3-pip libopenal1

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
