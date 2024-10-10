# syntax=docker/dockerfile:1

# base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base
SHELL ["/bin/bash", "-c"]

# build args and environment variables
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG CUBE_ROTATION_OBELISK_ROOT=/
ARG DEV=false

ENV USER=$USER
ENV UID=$UID
ENV GID=$GID
ENV CUBE_ROTATION_OBELISK_ROOT=$CUBE_ROTATION_OBELISK_ROOT
ENV DEV=$DEV

ENV XDG_RUNTIME_DIR=/run/user/${UID}

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# basic dependencies
# everything below sudo is required for this project - everything above is required for Obelisk compatibility
# git-lfs: for downloading large files from git
# libccd-dev: mujoco simulation dependency
# libxinerama-dev, libxcursor-dev, libxi-dev: mujoco visualization dependencies
# libboost-all-dev: custom rviz2 plugin dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    mesa-common-dev \
    locales \
    sudo \
    git-lfs \
    libccd-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libboost-all-dev \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# create non-root user with sudo privileges
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash && \
    echo "${USER}:password" | chpasswd && \
    usermod -aG sudo ${USER} && \
    echo "%${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# switch to new user and workdir
USER ${UID}

# download and untar pre-built mujoco binaries
# The main reason we need to do this is that mjpc and Obelisk have strange conflicts in the build process - if I point
# Obelisk to a completely separate directory that has nothing to do with the build dir in the workspace, things work.
ENV MUJOCO_VERSION=3.2.0
ENV MUJOCO_SOURCE_DIR=/home/${USER}/mujoco_source_dir
RUN wget -P /home/${USER} https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf /home/${USER}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -C /home/${USER} && \
    rm /home/${USER}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz && \
    mv /home/${USER}/mujoco-${MUJOCO_VERSION}/ ${MUJOCO_SOURCE_DIR}

# clone Obelisk into this docker image at the $OBELISK_ROOT path
RUN git clone https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk && \
    cd /home/${USER}/obelisk && \
    git checkout 20a1af91ef108239f33a0224318bbdb8170429ca
ENV OBELISK_ROOT=/home/${USER}/obelisk

# set up Obelisk in the Docker container
# the installation should occur in a non-root-owned directory for correct permissions during zed installer
WORKDIR /home/${USER}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs:/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH
RUN source /home/${USER}/obelisk/setup.sh \
    --install-sys-deps \
    --config-groups \
    --cyclone-perf \
    --obk-aliases \
    --basic \
    --leap \
    --zed-ai \
    --mj-source-dir ${MUJOCO_SOURCE_DIR}
RUN source /home/${USER}/.bash_aliases && obk

# install python dependencies
ENV PATH="/home/${USER}/.local/bin:$PATH"
ADD pyproject.toml cro README.md ${CUBE_ROTATION_OBELISK_ROOT}/
RUN pip install --upgrade pip && \
    pip install -e ${CUBE_ROTATION_OBELISK_ROOT}[dev]

# add cro aliases to ~/.bash_aliases
ADD .cro_aliases /tmp/.cro_aliases
RUN cat /tmp/.cro_aliases >> /home/${USER}/.bash_aliases && \
    sudo rm /tmp/.cro_aliases
