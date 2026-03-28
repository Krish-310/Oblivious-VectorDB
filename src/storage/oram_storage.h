#ifndef ORAM_STORAGE_H
#define ORAM_STORAGE_H

// H2O2RAM
// Creates an ORAM-backed map that stores vectors obliviously.
// Requires storing data in RAM

#include "storage_adapter.h"

#include "../../include/H2O2RAM/include/omap.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnsw {
namespace storage {

// Allow 512 dimension vectors, for lower dims the remaining space is padded with zeros
static constexpr size_t ORAM_MAX_DIM = 512;

struct OramVectorBlock {
  float data[ORAM_MAX_DIM]; // vector payload — tail is zero-padded if dim < MAX_DIM
};

// -----------------------------------------------------------------------------
// OramStorage
//
// Implements StorageAdapter using H2O2RAM's O2Map as the backing store.
//
// Key design decisions:
//
//   1. get_vector() returns a pointer into a single-element local buffer
//      (ret_buf_). This is safe as long as the caller uses the pointer
//      before the next get_vector() call — which is always true in
//      HNSW's SEARCH-LAYER inner loop (it computes distance immediately).
//
//   2. The O2Map key is size_t (node ID). Value is OramVectorBlock.
//
//   3. Dimension is stored separately in trusted memory — it is not
//      secret (it's a public schema parameter).
// -----------------------------------------------------------------------------
class OramStorage : public StorageAdapter {
public:
  // Load all vectors into the O2Map at construction time.
  OramStorage(const float *raw_vectors, size_t n, size_t dim)
      : dim_(dim), n_(n)
  {
    if (dim == 0)
      throw std::invalid_argument("OramStorage: dim must be > 0");
    if (dim > ORAM_MAX_DIM)
      throw std::invalid_argument("OramStorage: dim " + std::to_string(dim) +
                                  " exceeds ORAM_MAX_DIM " +
                                  std::to_string(ORAM_MAX_DIM) +
                                  ". Increase ORAM_MAX_DIM and recompile.");
    if (raw_vectors == nullptr && n > 0)
      throw std::invalid_argument("OramStorage: null raw_vectors");

    OramVectorBlock block;
    std::memset(&block, 0, sizeof(block)); // zero-pad unused tail

    for (size_t id = 0; id < n; ++id) {
      const float *src = raw_vectors + id * dim;
      std::memcpy(block.data, src, dim * sizeof(float));
      omap_.insert(static_cast<uint32_t>(id), block);
    }
  }

  // -------------------------------------------------------------------------
  // get_vector(id)
  //
  // Oblivious read: the O2Map touches the same number of hash-table buckets
  // regardless of which id is requested, hiding which vector is being fetched.
  //
  // Uses a round-robin pool of 64 return buffers to handle HNSW's pruning
  // heuristic which holds up to M_max0 (typically 32) live pointers at once.
  // Without this, a single-buffer implementation would alias and silently cause
  // every pruning distance to evaluate to 0.
  // -------------------------------------------------------------------------
  const float *get_vector(size_t id) override {
    if (id >= n_)
      throw std::out_of_range("OramStorage::get_vector: id " +
                              std::to_string(id) + " out of range [0, " +
                              std::to_string(n_) + ")");

    // Use a buffer slot to return
    current_buf_idx_ = (current_buf_idx_ + 1) % NUM_BUFFERS;
    // Copy-assign from ORAM reference into our owned buffer slot.
    ret_bufs_[current_buf_idx_] = omap_[static_cast<uint32_t>(id)];
    return ret_bufs_[current_buf_idx_].data;
  }

  size_t get_dim() const override { return dim_; }

  // Oblivious Write - Good to have
  void put_vector(size_t id, const float *vec) {
    OramVectorBlock block;
    std::memset(&block, 0, sizeof(block));
    std::memcpy(block.data, vec, dim_ * sizeof(float));
    omap_.insert(static_cast<uint32_t>(id), block);
  }

  size_t size() const { return n_; }

private:
  size_t dim_;
  size_t n_;

  // The ObliviousMap: key = uint32_t node ID, value = OramVectorBlock
  // H2O2RAM's ORAM-backed map hides which vector is being accessed.
  ORAM::ObliviousMap<uint32_t, OramVectorBlock> omap_;

  // Round-robin pool of 64 return buffers.
  // HNSW's SELECT-NEIGHBORS heuristic holds up to M_max0 (~32) live
  // get_vector() pointers simultaneously (one per already-selected neighbor).
  // 64 slots gives a safe 2× margin over that maximum.
  // Memory cost: 64 × 512 floats × 4 bytes = 128 KB (negligible).
  static constexpr int NUM_BUFFERS = 64;
  OramVectorBlock ret_bufs_[NUM_BUFFERS];
  uint8_t current_buf_idx_ = 0;
};

} // namespace storage
} // namespace hnsw

#endif // ORAM_STORAGE_H