#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H

#include "lru_cache.h"
#include "storage_adapter.h"
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace hnsw {
namespace storage {

class DiskStorage : public StorageAdapter {
public:
  // cache_capacity: number of vectors to cache in RAM
  DiskStorage(const std::string &filepath, size_t dim,
              size_t cache_capacity = 10000)
      : dim_(dim), cache_(cache_capacity, dim), fd_(-1) {

    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("Failed to open file: " + filepath);
    }
  }

  ~DiskStorage() override {
    if (fd_ != -1) {
      close(fd_);
    }
  }

  const float *get_vector(size_t id) override {
    // 1. Check if it's in the cache
    const float *cached_vec = cache_.get(id);
    if (cached_vec != nullptr) {
      return cached_vec;
    }

    // 2. Not in cache, read from disk using pread
    // Add 8 bytes offset for the header (num_elements, dim)
    std::vector<float> buffer(dim_);
    off_t offset = 8 + static_cast<off_t>(id) * dim_ * sizeof(float);

    ssize_t bytes_read =
        pread(fd_, buffer.data(), dim_ * sizeof(float), offset);
    if (bytes_read != static_cast<ssize_t>(dim_ * sizeof(float))) {
      throw std::runtime_error("Failed to read vector from disk at id: " +
                               std::to_string(id));
    }

    // 3. Put into cache and return the cached pointer
    return cache_.put(id, buffer);
  }

  size_t get_dim() const override { return dim_; }

private:
  size_t dim_;
  LRUCache cache_;
  int fd_;
};

} // namespace storage
} // namespace hnsw

#endif // DISK_STORAGE_H
