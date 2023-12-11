FROM continuumio/miniconda3:latest

WORKDIR /vizdoom

RUN conda install -y -c openal-soft

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
