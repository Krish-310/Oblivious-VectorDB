#ifndef MEMORY_STORAGE_H
#define MEMORY_STORAGE_H

#include "storage_adapter.h"

namespace hnsw {
namespace storage {

class MemoryStorage : public StorageAdapter {
public:
  MemoryStorage(const float *raw_data, size_t dim)
      : data_ptr_(raw_data), dim_(dim) {}

  const float *get_vector(size_t id) override { return data_ptr_ + id * dim_; }

  size_t get_dim() const override { return dim_; }

private:
  const float *data_ptr_;
  size_t dim_;
};

} // namespace storage
} // namespace hnsw

#endif // MEMORY_STORAGE_H
