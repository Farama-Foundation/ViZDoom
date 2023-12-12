FROM continuumio/miniconda3:latest

WORKDIR /vizdoom

RUN apt update && apt install -y libopenal1

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
