#include "hnsw.h"
#include "distance.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "../../include/H2O2RAM/include/omap.hpp"

#include "../utils/oblivious_heap.h"

namespace oblivious_hnsw {

HNSW::HNSW(size_t dim, int max_elements, int M, int ef_construction,
           DistFunc dist_func)
    : dim_(dim), max_elements_(max_elements), M_(M),
      ef_construction_(ef_construction), dist_func_(dist_func) {
  M0_ = M * 2;
  mult_ = 1.0 / log(1.0 * M);
  max_level_ = -1;
  enterpoint_node_ = -1;
  num_elements_ = 0;

  level_generator_.seed(100);

  // Pre-allocate the arrays since we know the maximum number of elements
  node_level_.resize(max_elements_, -1);

  // Create 20 empty layers as a safe maximum for random level generation
  graph_.resize(20);

  // Initialize the visited tag logic
  visited_tag_ = 0;

  // Pre-populate the visited ObliviousMap with all possible node IDs (plus
  // DUMMY) so that operator[] can overwrite them in-place during search without
  // triggering unbounded ORAM exponential bounds resizing!
  for (int i = 0; i < max_elements_; i++) {
    visited_.insert(i, 0);
  }
  visited_.insert(static_cast<uint32_t>(-1), 0);
}

HNSW::~HNSW() {
  // Cleanup allocated memory for graphs when implemented
}

int HNSW::get_random_level() {
  double f = uniform_dist_(level_generator_);
  if (f == 0.0) {
    f = 1e-6;
  }
  return (int)(-log(f) * mult_);
}

void HNSW::insert(int label, const float *vector) {

  utils::ObliviousMaxHeap W;
  int ep = enterpoint_node_;
  int L = max_level_;
  int l = get_random_level();

  // Make sure we have enough layers allocated if we roll a very high random
  // level
  if (l >= (int)graph_.size()) {
    graph_.resize(l + 1);
  }

  // Save the maximum level of this new node
  node_level_[label] = l;

  // Explicitly initialize an empty node at all active layers for this label
  // Since graph_ is now an ObliviousMap, we cannot rely on vector
  // pre-allocation.
  OramGraphNode empty_node;
  empty_node.count = 0;
  for (int layer = 0; layer <= l; layer++) {
    graph_[layer].insert(label, empty_node);
  }

  if (ep == -1) {
    enterpoint_node_ = label;
    max_level_ = l;
    num_elements_++;
    return;
  }

  // Calculate distance to entrypoint to start the search
  float dist_ep = dist_func_(vector, ep, storage_, dim_);

  for (int l_c = L; l_c > l; l_c--) {
    int T_bound = (l_c == 0) ? T0_build_ : T_build_;
    W = _search_layer(vector, ep, 1, l_c, T_bound);

    // Convert the max-heap W to find the nearest valid element securely
    // Since ef=1, we can isolate it by popping dummy values out
    int next_ep = ep;
    float next_min_dist = std::numeric_limits<float>::infinity();
    while (!W.empty()) {
      dist_pair p = W.pop_max();
      if (p.second != -1 && p.first < next_min_dist) {
        next_min_dist = p.first;
        next_ep = p.second;
      }
    }
    ep = next_ep;
  }

  for (int l_c = std::min(L, l); l_c >= 0; l_c--) {
    int T_bound = (l_c == 0) ? T0_build_ : T_build_;
    W = _search_layer(vector, ep, ef_construction_, l_c, T_bound);
    std::vector<int> neighbours = _select_neighbours(vector, W, M_, l_c);

    // add bidirectional connections
    for (int neighbor : neighbours) {
      OramGraphNode node_label = graph_[l_c][label];
      if (node_label.count < ORAM_MAX_EDGES) {
        node_label.edges[node_label.count++] = neighbor;
        graph_[l_c][label] = node_label;
      }

      OramGraphNode node_neighbor = graph_[l_c][neighbor];
      if (node_neighbor.count < ORAM_MAX_EDGES) {
        node_neighbor.edges[node_neighbor.count++] = label;
        graph_[l_c][neighbor] = node_neighbor;
      }

      // shrink connections if needed using a simple furthest-node heuristic
      if (node_neighbor.count > ((l_c == 0) ? M0_ : M_)) {
        const float *n_vector = storage_->get_vector(neighbor);

        float max_dist = -1.0;
        int furthest_idx = -1;

        for (int i = 0; i < node_neighbor.count; i++) {
          int candidate = node_neighbor.edges[i];
          float d = dist_func_(n_vector, candidate, storage_, dim_);
          if (d > max_dist) {
            max_dist = d;
            furthest_idx = i;
          }
        }

        // Remove the furthest connection sequentially
        if (furthest_idx != -1) {
          for (int i = furthest_idx; i < node_neighbor.count - 1; i++) {
            node_neighbor.edges[i] = node_neighbor.edges[i + 1];
          }
          node_neighbor.count--;
          graph_[l_c][neighbor] = node_neighbor;
        }
      }

      // Also shrink the newly inserted node's connections if needed
      if (node_label.count > ((l_c == 0) ? M0_ : M_)) {
        const float *n_vector = storage_->get_vector(label);

        float max_dist = -1.0;
        int furthest_idx = -1;

        for (int i = 0; i < node_label.count; i++) {
          int candidate = node_label.edges[i];
          float d = dist_func_(n_vector, candidate, storage_, dim_);
          if (d > max_dist) {
            max_dist = d;
            furthest_idx = i;
          }
        }

        // Remove the furthest connection
        if (furthest_idx != -1) {
          for (int i = furthest_idx; i < node_label.count - 1; i++) {
            node_label.edges[i] = node_label.edges[i + 1];
          }
          node_label.count--;
          graph_[l_c][label] = node_label;
        }
      }
    }

    // Check the remaining elements in W to find the minimal route for layer
    // descent
    int nearest_node = ep;
    float min_dist = std::numeric_limits<float>::infinity();
    while (!W.empty()) {
      dist_pair p = W.pop_max();
      bool is_valid = (p.second != -1) && (p.first < min_dist);
      min_dist = is_valid ? p.first : min_dist;
      nearest_node = is_valid ? p.second : nearest_node;
    }
    ep = nearest_node;
  }

  if (l > L) {
    enterpoint_node_ = label;
    max_level_ = l;
  }
  num_elements_++;
}

std::vector<int> HNSW::_select_neighbours(const float *query,
                                          utils::ObliviousMaxHeap candidates,
                                          int M, int level) {
  // Simple heuristic: just take the closest M elements from the candidates
  std::vector<int> res;
  res.reserve(M);

  // candidates is a max-heap (largest distance at top)
  // We want the smallest distance ones at the front of the result array.
  // First, pop them all into a vector. They will be ordered largest to
  // smallest.
  std::vector<dist_pair> sorted_candidates;
  while (!candidates.empty()) {
    dist_pair p = candidates.pop_max();
    if (p.second != -1) { // Ignore DUMMY elements
      sorted_candidates.push_back(p);
    }
  }

  // Then iterate backwards from smallest to largest
  for (int i = sorted_candidates.size() - 1; i >= 0 && res.size() < (size_t)M;
       i--) {
    res.push_back(sorted_candidates[i].second);
  }

  return res;
}

utils::ObliviousMaxHeap HNSW::_search_layer(const float *query, int ep, int ef,
                                            int level, int T) {

  visited_tag_++;
  if (visited_tag_ == 0) {
    visited_tag_ = 1;
  }

  utils::ObliviousMinHeap candidates;
  utils::ObliviousMaxHeap found;

  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr int DUMMY = -1;

  dist_pair start_node = {dist_func_(query, ep, storage_, dim_), ep};
  found.insert(start_node.first, start_node.second);
  for (int i = 0; i < ef; i++) {
    found.insert(inf, DUMMY);
  }

  candidates.insert(start_node.first, start_node.second);
  visited_[ep] = visited_tag_;

  // T outer loop iterations passed natively from the caller context bounds
  for (int i = 0; i < T; i++) {

    dist_pair c = candidates.pop_min();

    int level_connections = (level > 0) ? M_ : M0_;

    // PREVENT EXCEPTION:
    // Route DUMMY to index 0 dynamically. Since `c.second` being DUMMY flips
    // `is_dummy` to true, all resulting nodes parsed from Node 0 will be safely
    // ignored natively downstream!
    uint32_t safe_c = (c.second == DUMMY) ? 0 : c.second;
    OramGraphNode c_neighbours = graph_[level][safe_c];

    for (int i = 0; i < level_connections; i++) {

      int e = c_neighbours.edges[i];

      // PREVENT EXCEPTION: Mask out DUMMY to 0 for the memory fetch
      int safe_e = (e == DUMMY) ? 0 : e;
      float e_dist = dist_func_(query, safe_e, storage_, dim_);

      bool seen = (visited_[safe_e] == visited_tag_);
      bool is_dummy = (e == DUMMY) || (c.second == DUMMY);

      bool add_to_visited = (!seen && !is_dummy);

      // oblivious update of visited set in-place
      uint32_t target_e = add_to_visited ? e : DUMMY;
      unsigned int target_tag = add_to_visited ? visited_tag_ : 0;
      visited_[target_e] = target_tag;

      // Use the actual distance from the found heap to preserve infinity
      // bounds!
      float f2_dist = found.get_max().first;

      bool better = (e_dist < f2_dist);

      bool do_insert = (add_to_visited && better);

      float target_dist = do_insert ? e_dist : inf;
      int target_node = do_insert ? e : DUMMY;
      found.insert(target_dist, target_node);
      found.pop_max();

      candidates.insert(target_dist, target_node);
    }
  }

  return found;
}

std::vector<int> HNSW::search(const float *query, int k, int ef_search) {
  if (enterpoint_node_ == -1) {
    return std::vector<int>(); // Empty graph
  }

  int ep = enterpoint_node_;
  int L = max_level_;

  // Phase 1: Descend through upper layers until layer 1
  for (int l_c = L; l_c > 0; l_c--) {
    int T_bound = (l_c == 0) ? T0_ : T_;
    utils::ObliviousMaxHeap W = _search_layer(query, ep, 1, l_c, T_);

    // Find the valid node locally filtering against dummies
    int next_ep = ep;
    float min_d = std::numeric_limits<float>::infinity();
    while (!W.empty()) {
      dist_pair p = W.pop_max();

      bool is_valid = (p.second != -1) && (p.first < min_d);

      min_d = is_valid ? p.first : min_d;
      next_ep = is_valid ? p.second : next_ep;
    }
    ep = next_ep;
  }

  // Phase 2: Search at bottom layer (0) with ef_search
  utils::ObliviousMaxHeap W = _search_layer(query, ep, ef_search, 0, T0_);

  // Return the nearest `k` candidates from W
  return _select_neighbours(query, W, k, 0);
}

void HNSW::save_index(const std::string &filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file for writing index: " + filepath);
  }

  out.write((char *)&enterpoint_node_, sizeof(int));
  out.write((char *)&max_level_, sizeof(int));
  out.write((char *)&num_elements_, sizeof(int));

  out.write((char *)node_level_.data(), max_elements_ * sizeof(int));

  int num_layers = max_level_ + 1;
  out.write((char *)&num_layers, sizeof(int));

  for (int l = 0; l < num_layers; l++) {
    for (int i = 0; i < max_elements_; i++) {
      OramGraphNode node = graph_[l][i];
      int sz = node.count;
      out.write((char *)&sz, sizeof(int));
      if (sz > 0) {
        out.write((char *)node.edges, sz * sizeof(int));
      }
    }
  }
}

void HNSW::load_index(const std::string &filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file for reading index: " + filepath);
  }

  in.read((char *)&enterpoint_node_, sizeof(int));
  in.read((char *)&max_level_, sizeof(int));
  in.read((char *)&num_elements_, sizeof(int));

  in.read((char *)node_level_.data(), max_elements_ * sizeof(int));

  int num_layers;
  in.read((char *)&num_layers, sizeof(int));

  if (num_layers > (int)graph_.size()) {
    graph_.resize(num_layers);
  }

  for (int l = 0; l < num_layers; l++) {
    for (int i = 0; i < max_elements_; i++) {
      int sz;
      in.read((char *)&sz, sizeof(int));
      if (sz > 0) {
        OramGraphNode node;
        node.count = sz;
        in.read((char *)node.edges, sz * sizeof(int));
        graph_[l].insert(i, node);
      }
    }
  }
}

} // namespace oblivious_hnsw
