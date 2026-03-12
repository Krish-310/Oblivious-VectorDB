#!/usr/bin/env bash
# ==============================================================================
# run_docker.sh
# 
# Helper script to quickly build and drop into the Oblivious VectorDB
# x86_64 Docker testing environment.
# ==============================================================================

set -e

IMAGE_NAME="oblivious-vectordb"

if [[ "$(docker images -q "$IMAGE_NAME" 2> /dev/null)" == "" ]]; then
    echo "========================================"
    echo "[1/2] Building Docker Image..."
    echo "========================================"
    # Build the image using the local Dockerfile
    # --platform=linux/amd64 is forced in the Dockerfile
    docker build -t "$IMAGE_NAME" .
else
    echo "========================================"
    echo "[1/2] Docker Image '$IMAGE_NAME' already exists."
    echo "      Skipping build step. To rebuild, run:"
    echo "      docker rmi $IMAGE_NAME"
    echo "========================================"
fi

echo ""
echo "========================================"
echo "[2/2] Launching Interactive Session..."
echo "========================================"
echo "-> Mounting ./data/ to /app/data/"
echo "-> You are now inside the x86_64 container."
echo ""
echo "To compile and run your ORAM VectorDB, type:"
echo "    make clean && make -j"
echo "    ./hnsw_eval ./data/"
echo "========================================"

# Run the container interactively, automatically deleting it on exit
# Mount the local ./data directory into the container
docker run -it --rm \
    -v "$(pwd):/app" \
    "$IMAGE_NAME"
