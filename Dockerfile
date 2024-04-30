FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Metadata as described in the documentation
LABEL maintainer="mariotrivinor@gmail.com" \
      version="1.0" \
      description="Image with CUDA 12.1.0, pytorch, and other dependencies."

# Arguments to build Docker Image using CUDA
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA True

ENV CUDA_HOME /usr/local/cuda/
ENV DEBIAN_FRONTEND=noninteractive
USER root

# Install essential packages and Jupyter Notebook
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-opencv \
    wget \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch, torchvision, and torchaudio with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y git

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y mpich libmpich-dev
WORKDIR /wd
COPY requirements.txt /wd
RUN pip install -r requirements.txt

# Copy your directories and set the working directory
WORKDIR /wd

RUN apt-get update && apt-get install -y curl

# Set permissions
RUN chmod -R 777 /wd

# Create a user with UID 1000 and GID 1000
RUN useradd -m -s /bin/bash -N -u 1000 jovyan

# Switch to this user
USER jovyan

