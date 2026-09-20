FROM continuumio/miniconda3:latest

WORKDIR /vizdoom

RUN conda install -y -c conda-forge gcc gxx rhash make cmake boost sdl3 openal-soft

COPY . ./
CMD ["bash", "./scripts/build_and_test_conda.sh"]
