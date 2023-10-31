FROM continuumio/miniconda3:latest

WORKDIR /vizdoom

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
