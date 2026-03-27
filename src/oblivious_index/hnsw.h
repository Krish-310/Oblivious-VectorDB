#ifndef OBLIVIOUS_HNSW_H
#define OBLIVIOUS_HNSW_H

#include "../storage/storage_adapter.h"
#include "distance.h"
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "../../include/H2O2RAM/include/omap.hpp"
#include "../utils/oblivious_heap.h"

namespace oblivious_hnsw {

// Since ORAM blocks require a compile-time fixed layout, we pad the edges.
// Adjust MAX_EDGES if you intend to run M0 > 64.
static constexpr int ORAM_MAX_EDGES = 64;

struct OramGraphNode {
    int count;
    int edges[ORAM_MAX_EDGES];

    OramGraphNode() : count(0) {
        std::fill_n(edges, ORAM_MAX_EDGES, -1);
    }
};

// A simple max-heap element where 'first' is distance, 'second' is node id
using dist_pair = std::pair<float, int>;

class HNSW {
public:
  using DistFunc = float (*)(const float *, int, hnsw::storage::StorageAdapter *, size_t);

  // Initialize the HNSW index structure
  HNSW(size_t dim, int max_elements, int M = 16, int ef_construction = 200,
       DistFunc dist_func = L2Sqr);
  ~HNSW();

  // Insert a new vector into the index
  void insert(int label, const float *vector);

  // Serialize index to disk
  void save_index(const std::string &filepath);

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

  // Oblivious Traversal
  int T_ = 20;
  int T0_ = 40;

  // Build constraints (Isolated to prevent search query dynamic scaling)
  int T_build_ = 30;
  int T0_build_ = 200;

  std::mt19937 level_generator_;
  std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};

  int get_random_level();

  utils::ObliviousMaxHeap _search_layer(const float *query, int ep,
                                               int ef, int level, int T);
  std::vector<int> _select_neighbours(const float *query,
                                      utils::ObliviousMaxHeap candidates,
                                      int M, int level);

  DistFunc dist_func_;

  // --- SIMPLE ADJACENCY LIST GRAPH STRUCTURE --- //

  // graph_[L][N] gives the vector of neighbor IDs for node N on layer L.
  // We allocate this up to some maximum expected number of levels (e.g., 20).
  // graph_[L] evaluates to an ObliviousMap linking NodeID -> OramGraphNode
  std::vector<ORAM::ObliviousMap<uint32_t, OramGraphNode>> graph_;

  // We still need to know the max level each node exists on
  std::vector<int> node_level_;

  // High performance visited map to obliviously track query traversals
  ORAM::ObliviousMap<uint32_t, unsigned int> visited_;
  unsigned int visited_tag_;

  // Pointer to the storage layer, which handles memory or disk vectors
  hnsw::storage::StorageAdapter *storage_ = nullptr;

public:
  void set_storage(hnsw::storage::StorageAdapter *storage) { storage_ = storage; }
};

} // namespace oblivious_hnsw

#endif // OBLIVIOUS_HNSW_H
