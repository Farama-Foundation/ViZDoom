FROM fedora:latest

WORKDIR vizdoom

# MINIMAL
RUN dnf update -y && dnf clean all && dnf install -y python3-devel python3-pip

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
