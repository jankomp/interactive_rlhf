# Based on OpenAI's mujoco-py Dockerfile

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    ffmpeg \
    git \
    git-lfs \
    ssh \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    parallel \
    python3.8 \
    python3.8-dev \
    python3-pip \
    rsync \
    software-properties-common \
    unzip \
    vim \
    virtualenv \
    xpra \
    xserver-xorg-dev \
    patchelf  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# for rendering
RUN apt-get update && apt-get install -y xvfb

ENV LANG C.UTF-8

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Run Xdummy mock X server by default so that rendering will work.
COPY ci/xorg.conf /etc/dummy_xorg.conf
COPY ci/Xdummy-entrypoint.py /usr/bin/Xdummy-entrypoint.py
ENTRYPOINT ["/usr/bin/Xdummy-entrypoint.py"]

# python-req stage contains Python venv, but not code.
# It is useful for development purposes: you can mount
# code from outside the Docker container.
FROM base as python-req

WORKDIR /imitation
# Copy over just setup.py and dependencies (__init__.py and README.md)
# to avoid rebuilding venv when requirements have not changed.
COPY ./setup.py ./setup.py
COPY ./README.md ./README.md
COPY ./src/imitation/__init__.py ./src/imitation/__init__.py
COPY ci/build_and_activate_venv.sh ./ci/build_and_activate_venv.sh

# Copy the .git directory into the Docker image
COPY .git .git

# Now setuptools-scm can determine the version of the imitation package correctly
RUN ci/build_and_activate_venv.sh /venv \
    && rm -rf $HOME/.cache/pip

# full stage contains everything.
# Can be used for deployment and local testing.
FROM python-req as full

# Delay copying (and installing) the code until the very end
COPY . /imitation
# install whole directory
RUN pip install -e .
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# RUN python3 setup.py sdist bdist_wheel
# RUN pip install --upgrade dist/imitation-*.whl

# for using reacher
RUN pip install gymnasium[mujoco]

# Fix swimmer environment
COPY src/imitation/util/custom_envs/swimmer.xml /venv/lib/python3.8/site-packages/gymnasium/envs/mujoco/assets/swimmer.xml

# for presenting the videos when gathering preferences from humans
RUN pip install opencv-python pygame moviepy flask flask_cors fastdtw imageio


# Default entrypoints
CMD ["/bin/bash"]

# HOW TO RUN JUPYTER NOTEBOOK FOR RENDERING VIDEO
# IN THE DOCKER CONTAINER SHELL:
# $ cd docs/tutorials
# $ xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip 0.0.0.0 --port 5000 --no-browser --allow-root --NotebookApp.token=''