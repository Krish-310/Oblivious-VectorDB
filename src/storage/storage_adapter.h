#ifndef STORAGE_ADAPTER_H
#define STORAGE_ADAPTER_H

#include <cstddef>

namespace hnsw {
namespace storage {

class StorageAdapter {
public:
  virtual ~StorageAdapter() = default;

  // Retrieve a pointer to a vector by its integer ID
  virtual const float *get_vector(size_t id) = 0;

  // Return the dimensionality of the vectors in this storage
  virtual size_t get_dim() const = 0;
};

} // namespace storage
} // namespace hnsw

#endif // STORAGE_ADAPTER_H
