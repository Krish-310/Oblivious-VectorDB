# Oblivious-VectorDB (HNSW C++)

This project is a clean-room C++ implementation of the Hierarchical Navigable Small World (HNSW) graph algorithm for approximate nearest neighbor search. It features both a standard memory-efficient implementation and an **experimental oblivious variant** utilizing H2O2RAM primitives designed for deployment within Trusted Execution Environments (TEEs) like Intel SGX.

## 1. Setup & Data Preparation

First, download and format a dataset from [ann-benchmarks](http://ann-benchmarks.com/). Since newer macOS versions restrict global package installations, we recommend using a virtual environment:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install required packages
pip install numpy h5py

# 4. Download and convert a dataset (e.g., sift-128-euclidean)
python3 scripts/download_and_convert.py --dataset sift-128-euclidean --max-vectors 1000

# 5. Deactivate the virtual environment
deactivate
```

## 2. Building the Project

### Native Compilation (x86 Linux)
If you are natively on an x86_64 Linux machine, you can compile the project directly using the provided `Makefile`:

```bash
make -j
```

### Docker Compilation (Apple Silicon / macOS)
If you are on an Apple Silicon device or standard Mac, the H2O2RAM library's x86 AES-NI intrinsics will fail to compile natively. We provide an automated Docker script that builds an `amd64` Ubuntu container and drops you into an emulated bash shell:

```bash
# Start the Docker container
./run_docker.sh

# Inside the container shell, cleanly compile the project
make -j
```

## 3. Usage & CLI Options

Execute the built binary (`hnsw_eval`) to run the indexing and search evaluation. By default, it looks for the dataset in `./data/sift-128-euclidean`.

```bash
./hnsw_eval [dataset_path] [options]
```

### Options

* **`--index-type <standard|oblivious>`**: Selects the underlying HNSW graph implementation to use. 
  * `standard` (Default): Uses standard `std::vector` adjacency tracking. Fast and memory efficient.
  * `oblivious`: Uses `ORAM::ObliviousMap` and branchless `ObliviousHeap` arrays to securely mask memory access patterns and branching logic from host observability. Suitable for TEEs.
* **`--rebuild-index`**: Forces a fresh rebuild of the graph index from the dataset vectors, ignoring any previously saved `graph_edges_*.db` binaries in the dataset directory.

### Examples

**Run with the Standard Index:**
```bash
./hnsw_eval ./data/sift-128-euclidean --index-type standard
```

**Run with the Oblivious Index:**
```bash
./hnsw_eval ./data/sift-128-euclidean --index-type oblivious
```

**Force a Rebuild of the Oblivious Index on a Custom Dataset:**
```bash
./hnsw_eval ./data/glove-100-angular --index-type oblivious --rebuild-index
```

## 4. Project Structure

* `src/main.cpp` - Entrypoint, CLI parsing, and testing templates.
* `src/index/` - Standard core HNSW graph logic (`hnsw.h/.cpp`).
* `src/oblivious_index/` - Oblivious HNSW components with doubly-oblivious `OramGraphNode` logic.
* `src/storage/` - VectorDB abstraction layer `MemoryStorage` and mmap `DiskStorage` with `LRUCache`.
* `src/utils/` - Shared file I/O, linear scan heap wrappers (`oblivious_heap.cpp`), and timing utilities.
* `scripts/` - Python utilities for working with HDF5 benchmarks.
