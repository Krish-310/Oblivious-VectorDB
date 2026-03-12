# Oblivious VectorDB

A clean-room C++ implementation of **Hierarchical Navigable Small World (HNSW)** with pluggable oblivious storage backends — including a full **client-server ORAM** architecture using [H2O2RAM](https://github.com/55199789/H2O2RAM) for cryptographically oblivious vector access.

## Architecture

```
┌─────────────────────────────────────────────┐
│  CLIENT (hnsw_eval) — trusted               │
│  - HNSW search/build logic                  │
│  - H₂O₂RAM client state + AES-PRP keys      │
│  - Oblivious block lookup + rebuild          │
└──────────────────┬──────────────────────────┘
                   │  TCP: encrypted RPC calls
                   ▼
┌─────────────────────────────────────────────┐
│  SERVER (oram_server) — untrusted           │
│  - Stores encrypted block files per level   │
│  - Cannot decrypt, cannot correlate queries │
└─────────────────────────────────────────────┘
```

### Storage Modes

| Mode | Flag | Description |
|---|---|---|
| `disk` | `--storage disk` | mmap-backed DiskStorage (baseline) |
| `oram` | `--storage oram` | Local H2O2RAM ObliviousMap (in-process) |
| `network-oram` | `--server host:port` | Client-server ORAM over TCP |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy h5py

# Full 1M-vector dataset (standard):
python3 scripts/download_and_convert.py --dataset sift-128-euclidean

# Subsampled 100k-vector dataset (for ORAM memory budget):
python3 scripts/download_and_convert.py --dataset sift-128-euclidean --max-vectors 100000
```

## Building

Requires Docker on Apple Silicon — H2O2RAM uses x86-only Intel intrinsics.

```bash
./run_docker.sh    # builds linux/amd64 image, drops into bash shell
```

Inside the container:
```bash
make               # builds both hnsw_eval and oram_server
make client        # builds only hnsw_eval
make server        # builds only oram_server
```

## Running

### Baseline (disk storage)
```bash
./hnsw_eval --rebuild-index --storage disk
```

### Local ORAM (in-process, H2O2RAM)
```bash
./hnsw_eval --storage oram
```

### Client-Server ORAM (full split)
```bash
# Terminal 1 — start the block server:
./oram_server --port 7777 --data-dir ./block_store

# Terminal 2 — run client, connecting to server:
./hnsw_eval --server 127.0.0.1:7777
```

### Force index rebuild
```bash
./hnsw_eval --rebuild-index --storage disk
```

## Project Structure

```
src/
├── main.cpp                    # Client entrypoint + evaluation harness
├── index/
│   ├── hnsw.h / hnsw.cpp       # HNSW graph insertion + search
│   └── distance.h              # L2 / cosine distance functions
├── storage/
│   ├── storage_adapter.h       # Abstract StorageAdapter interface
│   ├── memory_storage.h        # RAM-backed storage
│   ├── disk_storage.h          # mmap-backed storage + LRU cache
│   ├── oram_storage.h          # Local H2O2RAM-backed storage
│   └── network_oram_storage.h  # Remote block server-backed storage
├── client/
│   └── oram_client.h           # Hierarchical ORAM client (PRF + oblivious rebuild)
├── server/
│   ├── block_store.h           # Flat encrypted block file manager
│   └── server_main.cpp         # TCP block server binary
├── shared/
│   └── protocol.h              # Binary RPC protocol (client ↔ server)
└── utils/
    └── utils.h                 # Binary I/O + timing utilities

third_party/H2O2RAM/            # H2O2RAM git submodule (oblivious primitives)
scripts/
└── download_and_convert.py     # Download + convert ann-benchmarks datasets
```
