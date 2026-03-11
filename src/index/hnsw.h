#ifndef HNSW_H
#define HNSW_H

#include "../storage/storage_adapter.h"
#include "distance.h"
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace hnsw {

// A simple max-heap element where 'first' is distance, 'second' is node id
using dist_pair = std::pair<float, int>;

class HNSW {
public:
  using DistFunc = float (*)(const float *, const float *, size_t);

  // Initialize the HNSW index structure
  HNSW(size_t dim, int max_elements, int M = 16, int ef_construction = 200,
       DistFunc dist_func = L2Sqr);
  ~HNSW();

  // Insert a new vector into the index
  void insert(int label, const float *vector);

  // Serialize index to disk
  void save_index(const std::string &filepath) const;

  // Load index from disk
  void load_index(const std::string &filepath);

  // Search for the k nearest neighbors to the query vector
  std::vector<int> search(const float *query, int k, int ef_search);

  // Optionally set the seed for reproducible random levels
  void set_seed(int seed) { level_generator_.seed(seed); }

private:
  size_t dim_;
  int max_elements_;
  int M_;
  int M0_; // Max connections for bottom layer
  int ef_construction_;

  double mult_; // Level multiplier -> 1 / ln(M)

  int max_level_;
  int enterpoint_node_;
  int num_elements_;

  std::mt19937 level_generator_;
  std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};

  int get_random_level();

  std::priority_queue<dist_pair> _search_layer(const float *query, int ep,
                                               int ef, int level);
  std::vector<int> _select_neighbours(const float *query,
                                      std::priority_queue<dist_pair> candidates,
                                      int M, int level);

  DistFunc dist_func_;

  // --- SIMPLE ADJACENCY LIST GRAPH STRUCTURE --- //

  // graph_[L][N] gives the vector of neighbor IDs for node N on layer L.
  // We allocate this up to some maximum expected number of levels (e.g., 20).
  std::vector<std::vector<std::vector<int>>> graph_;

  // We still need to know the max level each node exists on
  std::vector<int> node_level_;

  // High performance visited array to prevent allocations during search
  std::vector<unsigned int> visited_array_;
  unsigned int visited_tag_;

  // Pointer to the storage layer, which handles memory or disk vectors
  storage::StorageAdapter *storage_ = nullptr;

public:
  void set_storage(storage::StorageAdapter *storage) { storage_ = storage; }
};

} // namespace hnsw

#endif // HNSW_H
