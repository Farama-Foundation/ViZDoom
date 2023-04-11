FROM continuumio/miniconda3:latest

WORKDIR vizdoom

# MINIMAL
RUN conda install -c conda-forge c-compiler cxx-compiler make cmake boost sdl2 openal-soft

COPY . ./
CMD ["bash", "./scripts/build_and_test.sh"]
