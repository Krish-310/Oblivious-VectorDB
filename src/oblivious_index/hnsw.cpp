#include "hnsw.h"
#include "distance.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

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
  for (auto &layer_adj : graph_) {
    layer_adj.resize(max_elements_);
  }

  // Pre-allocate the visited array to avoid memory allocations during search
  visited_array_.resize(max_elements_, 0);
  visited_tag_ = 0;
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

  std::priority_queue<dist_pair> W;
  int ep = enterpoint_node_;
  int L = max_level_;
  int l = get_random_level();

  // Make sure we have enough layers allocated if we roll a very high random
  // level
  if (l >= (int)graph_.size()) {
    graph_.resize(l + 1);
    for (auto &layer_adj : graph_) {
      if (layer_adj.size() < (size_t)max_elements_) {
        layer_adj.resize(max_elements_);
      }
    }
  }

  // Save the maximum level of this new node
  node_level_[label] = l;

  // At this point, `graph_[layer][label]` is an empty std::vector<int> ready to
  // hold neighbors.

  if (ep == -1) {
    enterpoint_node_ = label;
    max_level_ = l;
    num_elements_++;
    return;
  }

  // Calculate distance to entrypoint to start the search
  float dist_ep = dist_func_(vector, ep, storage_, dim_);

  for (int l_c = L; l_c > l; l_c--) {
    W = _search_layer(vector, ep, 1, l_c);

    // Convert the max-heap W to find the nearest element
    // Since ef=1, W only has 1 element, but W.top() is the LARGEST (furthest)
    // For ef=1, it is both the furthest and nearest.
    ep = W.top().second;
  }

  for (int l_c = std::min(L, l); l_c >= 0; l_c--) {
    W = _search_layer(vector, ep, ef_construction_, l_c);
    std::vector<int> neighbours = _select_neighbours(vector, W, M_, l_c);

    // add bidirectional connections
    for (int neighbor : neighbours) {
      graph_[l_c][label].push_back(neighbor);
      graph_[l_c][neighbor].push_back(label);

      // shrink connections if needed using a simple furthest-node heuristic
      if (graph_[l_c][neighbor].size() > (size_t)((l_c == 0) ? M0_ : M_)) {
        int max_M = (l_c == 0) ? M0_ : M_;
        const float *n_vector = storage_->get_vector(neighbor);

        // Find the furthest connection to remove
        float max_dist = -1.0;
        int furthest_idx = -1;

        for (size_t i = 0; i < graph_[l_c][neighbor].size(); i++) {
          int candidate = graph_[l_c][neighbor][i];
          float d = dist_func_(n_vector, candidate, storage_, dim_);
          if (d > max_dist) {
            max_dist = d;
            furthest_idx = i;
          }
        }

        // Remove the furthest connection
        if (furthest_idx != -1) {
          graph_[l_c][neighbor].erase(graph_[l_c][neighbor].begin() +
                                      furthest_idx);
        }
      }

      // Also shrink the newly inserted node's connections if needed
      if (graph_[l_c][label].size() > (size_t)((l_c == 0) ? M0_ : M_)) {
        const float *n_vector = storage_->get_vector(label);

        float max_dist = -1.0;
        int furthest_idx = -1;

        for (size_t i = 0; i < graph_[l_c][label].size(); i++) {
          int candidate = graph_[l_c][label][i];
          float d = dist_func_(n_vector, candidate, storage_, dim_);
          if (d > max_dist) {
            max_dist = d;
            furthest_idx = i;
          }
        }

        if (furthest_idx != -1) {
          graph_[l_c][label].erase(graph_[l_c][label].begin() + furthest_idx);
        }
      }
    }

    // Convert max-heap to find the nearest (smallest distance) element in W for
    // the next layer down
    int nearest_node = ep;
    float min_dist = INFINITY;
    while (!W.empty()) {
      if (W.top().first < min_dist) {
        min_dist = W.top().first;
        nearest_node = W.top().second;
      }
      W.pop();
    }
    ep = nearest_node;
  }

  if (l > L) {
    enterpoint_node_ = label;
    max_level_ = l;
  }
  num_elements_++;
}

std::vector<int>
HNSW::_select_neighbours(const float *query,
                         std::priority_queue<dist_pair> candidates, int M,
                         int level) {
  // Simple heuristic: just take the closest M elements from the candidates
  std::vector<int> res;
  res.reserve(M);

  // candidates is a max-heap (largest distance at top)
  // We want the smallest distance ones at the front of the result array.
  // First, pop them all into a vector. They will be ordered largest to
  // smallest.
  std::vector<dist_pair> sorted_candidates;
  while (!candidates.empty()) {
    sorted_candidates.push_back(candidates.top());
    candidates.pop();
  }

  // Then iterate backwards from smallest to largest
  for (int i = sorted_candidates.size() - 1; i >= 0 && res.size() < (size_t)M;
       i--) {
    res.push_back(sorted_candidates[i].second);
  }

  return res;
}

std::priority_queue<dist_pair> HNSW::_search_layer(const float *query, int ep,
                                                   int ef, int level) {

  // v (visited nodes) using the zero-allocation tag method
  visited_tag_++;
  if (visited_tag_ == 0) { // Prevent overflow theoretically (4 billion queries)
    std::fill(visited_array_.begin(), visited_array_.end(), 0);
    visited_tag_ = 1;
  }
  visited_array_[ep] = visited_tag_;

  // C (candidate set) - min-heap to extract the closest node to query
  // Priority queue by default is max-heap, so we need greater to make it
  // min-heap
  auto cmp = [](const dist_pair &a, const dist_pair &b) {
    return a.first > b.first;
  };
  std::priority_queue<dist_pair, std::vector<dist_pair>, decltype(cmp)> C(cmp);

  // W (found nearest neighbors) - max-heap to keep the ef closest nodes
  std::priority_queue<dist_pair> W;

  // Calculate distance from query to entry point
  float dist_ep = dist_func_(query, ep, storage_, dim_);

  C.push({dist_ep, ep});
  W.push({dist_ep, ep});

  while (!C.empty()) {
    // Extract nearest element from C to query
    dist_pair curr = C.top();
    C.pop();

    // Get furthest element from W to query
    dist_pair furthest_W = W.top();

    if (curr.first > furthest_W.first) {
      break; // All elements in W are evaluated
    }

    // For each e in curr's neighbors at layer `level`
    for (int neighbor : graph_[level][curr.second]) {
      if (visited_array_[neighbor] != visited_tag_) {
        visited_array_[neighbor] = visited_tag_;

        float dist_neighbor = dist_func_(query, neighbor, storage_, dim_);

        furthest_W = W.top();

        if (dist_neighbor < furthest_W.first || W.size() < (size_t)ef) {
          C.push({dist_neighbor, neighbor});
          W.push({dist_neighbor, neighbor});

          if (W.size() > (size_t)ef) {
            W.pop();
          }
        }
      }
    }
  }

  return W;
}

std::vector<int> HNSW::search(const float *query, int k, int ef_search) {
  if (enterpoint_node_ == -1) {
    return std::vector<int>(); // Empty graph
  }

  int ep = enterpoint_node_;
  int L = max_level_;

  // Phase 1: Descend through upper layers until layer 1
  for (int l_c = L; l_c > 0; l_c--) {
    std::priority_queue<dist_pair> W = _search_layer(query, ep, 1, l_c);
    // W will only have the best node found at this layer because ef=1
    ep = W.top().second;
  }

  // Phase 2: Search at bottom layer (0) with ef_search
  std::priority_queue<dist_pair> W = _search_layer(query, ep, ef_search, 0);

  // Return the nearest `k` candidates from W
  return _select_neighbours(query, W, k, 0);
}

void HNSW::save_index(const std::string &filepath) const {
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
      int sz = graph_[l][i].size();
      out.write((char *)&sz, sizeof(int));
      if (sz > 0) {
        out.write((char *)graph_[l][i].data(), sz * sizeof(int));
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
    graph_[l].resize(max_elements_);
    for (int i = 0; i < max_elements_; i++) {
      int sz;
      in.read((char *)&sz, sizeof(int));
      graph_[l][i].resize(sz);
      if (sz > 0) {
        in.read((char *)graph_[l][i].data(), sz * sizeof(int));
      }
    }
  }
}

} // namespace oblivious_hnsw
