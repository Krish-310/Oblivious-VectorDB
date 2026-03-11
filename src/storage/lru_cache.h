#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <cstddef>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hnsw {
namespace storage {

class LRUCache {
public:
  LRUCache(size_t capacity, size_t dim) : capacity_(capacity), dim_(dim) {
    if (capacity_ == 0)
      throw std::invalid_argument("Capacity must be > 0");
  }

  // Returns pointer to cached vector, or nullptr if not in cache
  const float *get(size_t id) {
    auto it = map_.find(id);
    if (it == map_.end())
      return nullptr;

    // Move to front (most recently used)
    list_.splice(list_.begin(), list_, it->second);
    return it->second->data.data();
  }

  // Puts vector into cache, evicting if necessary
  const float *put(size_t id, const std::vector<float> &vec) {
    if (vec.size() != dim_)
      throw std::invalid_argument("Vector dimension mismatch");

    auto it = map_.find(id);
    if (it != map_.end()) {
      it->second->data = vec;
      list_.splice(list_.begin(), list_, it->second);
      return it->second->data.data();
    }

    if (list_.size() == capacity_) {
      size_t lru_id = list_.back().id;
      map_.erase(lru_id);
      list_.pop_back();
    }

    list_.push_front({id, vec});
    map_[id] = list_.begin();
    return list_.front().data.data();
  }

private:
  struct CacheNode {
    size_t id;
    std::vector<float> data;
  };

  size_t capacity_;
  size_t dim_;
  std::list<CacheNode> list_;
  std::unordered_map<size_t, std::list<CacheNode>::iterator> map_;
};

} // namespace storage
} // namespace hnsw

#endif // LRU_CACHE_H
