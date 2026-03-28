#ifndef OBLIVIOUS_DISTANCE_H
#define OBLIVIOUS_DISTANCE_H

#include <cstddef>

#include "../storage/storage_adapter.h"

namespace oblivious_hnsw {

// Basic L2 squared distance between two float vectors
inline float L2Sqr(const float *query, int target_id, hnsw::storage::StorageAdapter *storage, size_t dim) {
  const float *target = storage->get_vector(target_id);
  float res = 0;
  for (size_t i = 0; i < dim; ++i) {
    float t = query[i] - target[i];
    res += t * t;
  }
  return res;
}

// Inner product distance (often requested for text embeddings)
inline float InnerProduct(const float *query, int target_id, hnsw::storage::StorageAdapter *storage, size_t dim) {
  const float *target = storage->get_vector(target_id);
  float res = 0;
  for (size_t i = 0; i < dim; ++i) {
    res += query[i] * target[i];
  }
  // Often we want distance rather than similarity, so we return 1.0f - IP (if
  // normalized) Here we just return 1.0f - res as a placeholder logic.
  return 1.0f - res;
}

} // namespace oblivious_hnsw

#endif // OBLIVIOUS_DISTANCE_H
