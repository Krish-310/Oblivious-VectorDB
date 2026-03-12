# Use Ubuntu 22.04 with explicit linux/amd64 architecture
# This forces Docker Desktop on Mac to use Rosetta 2 emulation for x86_64
FROM --platform=linux/amd64 ubuntu:22.04

# Prevent interactive prompts blocking the build
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies required by H2O2RAM
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    libomp-dev \
    libssl-dev \
    libtbb-dev \
    libnlopt-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy the source code into the container
COPY . /app

# Default entrypoint to an interactive bash shell
CMD ["/bin/bash"]
