FROM fedora:latest

WORKDIR /vizdoom

# Install Python and pip
RUN dnf update -y && dnf clean all && dnf install -y python3-devel python3-pip openal-soft

COPY . ./
CMD ["bash", "./scripts/install_and_test_wheel.sh"]
