# Oblivious VectorDB Roadmap

## Phase 1: Baseline HNSW
- [x] **Basic HNSW Insert**
  - Implement heuristic neighbor selection (`select_neighbors_heuristic`).
  - Implement node insertion logic across levels.
  - Implement standard HNSW Search to verify baseline accuracy (Recall@K).

## Phase 2: VectorDB Foundation
- [x] **Data Storage Architecture**
  - Refactor `HNSW` class to interact via a `StorageAdapter` interface rather than direct array pointers.
  - Implement basic `MemoryStorage` (RAM-based) first.
  - (Optional) Implement `DiskStorage` (mmap-based) for handling datasets larger than RAM.

## Phase 3: Oblivious Graph Structures
- [ ] **Oblivious Priority Queue (PQ)**
  - Implement a basic Linear Oblivious Priority Queue for the candidate and found sets (`W` and `C`).
  - Evaluate the performance penalty compared to `std::priority_queue`.
- [ ] **Oblivious Graph Mapping**
  - Select and implement an oblivious map structure (e.g., Path ORAM or linear scan dictionary).
  - Modify the HNSW graph to store edge neighbor lists securely within the chosen oblivious map.

## Phase 4: Oblivious Search Implementation
- [ ] **Oblivious Search Algorithm**
  - Translate the standard HNSW search into the Oblivious HNSW search pseudocode.
  - Eliminate early-exit data dependencies and branch side-channels.
- [x] **Protect Vector Data Access**
  - Integrate H2O2RAM (Hardware-Assisted Oblivious RAM) to securely access the raw float vectors from the VectorDB storage layer during distance calculations.
  - Ensure distance computations (`distance.h`) do not leak memory access patterns.
