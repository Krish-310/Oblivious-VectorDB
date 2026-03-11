#ifndef DISTANCE_H
#define DISTANCE_H

#include <cstddef>

namespace hnsw {

// Basic L2 squared distance between two float vectors
inline float L2Sqr(const float *a, const float *b, size_t dim) {
  float res = 0;
  for (size_t i = 0; i < dim; ++i) {
    float t = a[i] - b[i];
    res += t * t;
  }
  return res;
}

// Inner product distance (often requested for text embeddings)
inline float InnerProduct(const float *a, const float *b, size_t dim) {
  float res = 0;
  for (size_t i = 0; i < dim; ++i) {
    res += a[i] * b[i];
  }
  // Often we want distance rather than similarity, so we return 1.0f - IP (if
  // normalized) Here we just return 1.0f - res as a placeholder logic.
  return 1.0f - res;
}

} // namespace hnsw

#endif // DISTANCE_H
