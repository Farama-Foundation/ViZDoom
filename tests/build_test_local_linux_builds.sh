#!/usr/bin/env bash
set -e

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'

DOCKERFILES_DIR=$( dirname ${BASH_SOURCE[0]} )/local_builds_dockerfiles
GENERATED_DOCKERFILES_DIR=tests/test_dockerfiles
IMAGE_PREFIX="vizdoom_local"

# Generate and run dockerfiles
# Array in format "<base docker image> <base dockerfile to use> <additional commands to add to the dockerfile after FROM statement>"
DOCKERFILES_TO_BUILD_AND_RUN=(
    "almalinux:9 dnf-based.Dockerfile RUN dnf install -y 'dnf-command(config-manager)' && dnf config-manager --set-enabled crb"
    "tgagor/centos:stream9 dnf-based.Dockerfile RUN dnf install -y 'dnf-command(config-manager)' && dnf config-manager --set-enabled crb"
    "fedora:36 dnf-based.Dockerfile"
    "fedora:37 dnf-based.Dockerfile"
    "rockylinux:9 dnf-based.Dockerfile RUN dnf install -y 'dnf-command(config-manager)' && dnf config-manager --set-enabled crb"
    "debian:11 apt-based.Dockerfile ENV LANG C.UTF-8"
    "debian:latest apt-based.Dockerfile ENV LANG C.UTF-8"
    "ubuntu:20.04 apt-based.Dockerfile"
    "ubuntu:22.04 apt-based.Dockerfile"
    "ubuntu:latest apt-based.Dockerfile"
    "ubuntu:20.04 apt+conda-based.Dockerfile"  # Ubuntu build with dependencies installed via conda
    #"continuumio/miniconda3:latest conda-based.Dockerfile"  # Does not work at the moment
)

# Test wheels inside docker containers
function create_dockerfile ( ) {
    local all_args=("$@")
    local base_image=$1
    local base_name=$( basename "$( echo ${base_image} | tr ':' '_' )" )
    local base_dockerfile=$2
    local add_commands=("${all_args[@]:2}")

    mkdir -p $GENERATED_DOCKERFILES_DIR
    dockerfile=${GENERATED_DOCKERFILES_DIR}/${IMAGE_PREFIX}_${base_name}.Dockerfile

    echo "FROM $base_image" > $dockerfile
    echo "" >> $dockerfile
    echo -e "${add_commands[@]}" >> $dockerfile
    cat ${DOCKERFILES_DIR}/$base_dockerfile | tail -n +2 >> $dockerfile
}

for dockerfile_setting in "${DOCKERFILES_TO_BUILD_AND_RUN[@]}"; do
    create_dockerfile $dockerfile_setting

    echo -n "Building and running $dockerfile, saving output to $dockerfile.log ... "
    filename=$( basename "$dockerfile" )
    dockerfile_dir=$( dirname "$dockerfile" )
    without_ext="${filename%.*}"
    tag="${without_ext}:latest"
    log="${dockerfile_dir}/${without_ext}.log"

    docker build -t $tag -f $dockerfile . &> $log || ( echo -e "${RED}FAILED${NC}"; exit 1 )
    docker run -it $tag &>> $log || ( echo -e "${RED}FAILED${NC}"; exit 1 )

    echo -e "${GREEN}OK${NC}"
done
