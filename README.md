# HNSW C++ Implementation

This project is a clean-room C++ implementation of the Hierarchical Navigable Small World (HNSW) graph algorithm for approximate nearest neighbor search.

## Setup

First, download and format a dataset from [ann-benchmarks](http://ann-benchmarks.com/). Since newer macOS versions restrict global package installations, we'll use a virtual environment:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install required packages
pip install numpy h5py

# 4. Download and convert the dataset (e.g., sift-128-euclidean)
python3 scripts/download_and_convert.py --dataset sift-128-euclidean

# 5. Deactivate the virtual environment
deactivate
```

## Building

A `Makefile` is provided. Run `make` to compile the project.

```bash
make
```

## Running

Execute the binary to run the evaluation. By default, it expects data to be located in `./data/sift-128-euclidean`.

```bash
./hnsw_eval
```

You can optionally pass a different path to the data directory:

```bash
```bash
./hnsw_eval ./data/glove-100-angular
```

To force a rebuild of the index (ignoring any saved `graph_edges.db`), use:
```bash
./hnsw_eval ./data/sift-128-euclidean --rebuild-index
```

## Structure

* `src/main.cpp` - Entrypoint testing script
* `src/index/` - Core Graph logic (`hnsw.h/.cpp`) and math metrics (`distance.h`)
* `src/storage/` - VectorDB abstraction layer `MemoryStorage` and mmap `DiskStorage` with `LRUCache`.
* `src/utils/` - Shared file I/O and timing utilities
* `scripts/` - Python utilities for working with HDF5 benchmarks
