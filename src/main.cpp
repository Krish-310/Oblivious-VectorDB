#include "index/hnsw.h"
#include "oblivious_index/hnsw.h"
#include "storage/disk_storage.h"
#include "storage/memory_storage.h"
#include "storage/oram_storage.h"
#include "utils/utils.h"
#include <iostream>
#include <string>
#include <vector>

struct Dataset {
  std::vector<float> base;
  std::vector<float> query;
  std::vector<int> ground_truth;
  int base_n, base_dim;
  int query_n, query_dim;
  int gt_n, gt_dim;
};

bool load_datasets(const std::string &data_path, Dataset &ds) {
  try {
    hnsw::read_binary_matrix(data_path + "/base.bin", ds.base, ds.base_n,
                             ds.base_dim);
    hnsw::read_binary_matrix(data_path + "/query.bin", ds.query, ds.query_n,
                             ds.query_dim);
    hnsw::read_binary_matrix(data_path + "/ground_truth.bin", ds.ground_truth,
                             ds.gt_n, ds.gt_dim);
  } catch (const std::exception &e) {
    std::cerr << "Error loading data: " << e.what() << "\n";
    std::cerr << "Did you run the download_and_convert.py script?\n";
    return false;
  }

  if (ds.base_dim != ds.query_dim) {
    std::cerr << "Dimension mismatch between base (" << ds.base_dim
              << ") and query (" << ds.query_dim << ")\n";
    return false;
  }

  std::cout << "[INFO] Data loaded successfully.\n";
  std::cout << "  -> Base points:  " << ds.base_n << "\n";
  std::cout << "  -> Query points: " << ds.query_n << "\n";
  std::cout << "  -> Dimensions:   " << ds.base_dim << "\n\n";
  return true;
}

template <typename IndexType>
void build_index(IndexType &index, const Dataset &ds,
                 const std::string &index_path) {
  std::cout << "========================================\n";
  std::cout << "[PHASE 1] Index Construction\n";
  std::cout << "========================================\n";

  hnsw::StopW timer;
  for (int i = 0; i < ds.base_n; ++i) {
    index.insert(i, ds.base.data() + i * ds.base_dim);
    if ((i + 1) % 100000 == 0) {
      std::cout << "  [Progress] Inserted " << (i + 1) << " / " << ds.base_n
                << " (" << ((float)(i + 1) / ds.base_n) * 100 << "%)\n";
    }
  }

  float build_time = timer.getElapsedTimeMicro() / 1e6;
  std::cout << "\n[RESULT] Index built successfully.\n";
  std::cout << "  -> Build Time:        " << build_time << " seconds\n";
  std::cout << "  -> Insertion Speed:   " << (ds.base_n / build_time)
            << " items/sec\n\n";

  std::cout << "  -> Saving index to " << index_path << "...\n";
  index.save_index(index_path);
  std::cout << "  -> Index saved successfully.\n\n";
}

template <typename IndexType>
void evaluate_search(IndexType &index, const Dataset &ds, int k,
                     int ef_search, int query_limit = -1) {
  std::cout << "========================================\n";
  std::cout << "[PHASE 2] Search Evaluation\n";
  std::cout << "========================================\n";
  std::cout << "Parameters:\n";
  std::cout << "  -> k (nearest neigbors): " << k << "\n";
  std::cout << "  -> ef_search:            " << ef_search << "\n";
  
  int queries_to_run = (query_limit > 0 && query_limit < ds.query_n) ? query_limit : ds.query_n;
  std::cout << "  -> Queries evaluated:    " << queries_to_run << " / " << ds.query_n << "\n\n";

  hnsw::StopW timer;
  int correct = 0;
  int total = 0; // Only count ground truth IDs actually in the indexed subset

  for (int i = 0; i < queries_to_run; ++i) {
    std::vector<int> results =
        index.search(ds.query.data() + i * ds.base_dim, k, ef_search);

    // Build set of achievable ground truth IDs (those actually indexed)
    std::vector<int> valid_truth;
    for (int truth_idx = 0;
         truth_idx < ds.gt_dim && (int)valid_truth.size() < k; ++truth_idx) {
      int gt_id = ds.ground_truth[i * ds.gt_dim + truth_idx];
      if (gt_id >= 0 && (size_t)gt_id < ds.base_n) {
        valid_truth.push_back(gt_id);
      }
    }
    total += (int)valid_truth.size();

    // Count how many results match an achievable ground truth ID
    for (int r : results) {
      for (int gt_id : valid_truth) {
        if (r == gt_id) {
          correct++;
          break;
        }
      }
    }
  }
  float search_time = timer.getElapsedTimeMicro() / 1e6;

  std::cout << "[RESULT] Search completed.\n";
  std::cout << "  -> Total Search Time: " << search_time << " seconds\n";
  std::cout << "  -> Queries Per Second: " << (queries_to_run / search_time)
            << " QPS\n";

  std::cout << "\n========================================\n";
  std::cout << "[FINAL METRICS]\n";
  std::cout << "========================================\n";
  if (total > 0) {
    std::cout << "  -> Achievable ground truth hits: " << total << " / "
              << (queries_to_run * k)
              << " (subset ratio: " << (100.0f * total / (queries_to_run * k))
              << "%)\n";
    std::cout << "  -> Accuracy (Recall@" << k
              << "): " << ((float)correct / total) * 100.0f << " %\n\n";
  } else {
    std::cout << "  -> Accuracy could not be computed (No achievable ground truth paths found).\n\n";
  }
}

template <typename IndexType>
void run_evaluation(const Dataset &ds, const std::string &data_path,
                    const std::string &storage_mode, const std::string &index_type, 
                    bool rebuild_index, int query_limit) {
  // HNSW Parameters
  int M = 16;
  int ef_construction = 200;

  std::cout << "Parameters:\n";
  std::cout << "  -> M (max edges):     " << M << "\n";
  std::cout << "  -> ef_construction:   " << ef_construction << "\n\n";

  IndexType index(ds.base_dim, ds.base_n, M, ef_construction);

  std::string index_path = data_path + "/graph_edges_" + index_type + ".db";
  bool index_loaded = false;

  if (!rebuild_index) {
    try {
      std::cout << "Attempting to load existing index from " << index_path
                << "...\n";
      index.load_index(index_path);
      std::cout << "Successfully loaded existing index.\n\n";
      index_loaded = true;
    } catch (...) {
      std::cout << "Index not found or invalid. Rebuilding from scratch...\n\n";
    }
  } else {
    std::cout
        << "Forced rebuild requested. Rebuilding index from scratch...\n\n";
  }

  if (!index_loaded) {
    hnsw::storage::MemoryStorage mem_storage(ds.base.data(), ds.base_dim);
    index.set_storage(&mem_storage);
    build_index(index, ds, index_path);
  }

  std::cout << "Storage Mode: " << storage_mode << "\n\n";

  int k = 10;
  int ef_search = 100;

  if (storage_mode == "disk") {
    hnsw::storage::DiskStorage disk_storage(data_path + "/base.bin",
                                            ds.base_dim, ds.base_n);
    index.set_storage(&disk_storage);
    evaluate_search(index, ds, k, ef_search, query_limit);
  } else {
    hnsw::storage::OramStorage oram_storage(ds.base.data(), ds.base_n,
                                            ds.base_dim);
    index.set_storage(&oram_storage);
    evaluate_search(index, ds, k, ef_search, query_limit);
  }
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0); // Disable stdout buffering
  std::string data_path = "./data/sift-128-euclidean";
  bool rebuild_index = false;
  std::string storage_mode = "oram"; // "oram" | "disk"
  std::string index_type = "standard"; // "standard" | "oblivious"
  int query_limit = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rebuild-index") {
      rebuild_index = true;
    } else if (arg == "--storage" && i + 1 < argc) {
      storage_mode = argv[++i];
    } else if (arg == "--num-queries" && i + 1 < argc) {
      query_limit = std::stoi(argv[++i]);
    } else if (arg == "--index-type" && i + 1 < argc) {
      index_type = argv[++i];
    } else {
      data_path = arg;
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "          HNSW C++ EVALUATION           \n";
  std::cout << "========================================\n\n";

  Dataset ds;
  if (!load_datasets(data_path, ds)) {
    return 1;
  }
  
  std::cout << "  -> Index Type:   " << index_type << "\n\n";

  if (index_type == "oblivious") {
    run_evaluation<oblivious_hnsw::HNSW>(ds, data_path, storage_mode, index_type, rebuild_index, query_limit);
  } else {
    run_evaluation<hnsw::HNSW>(ds, data_path, storage_mode, index_type, rebuild_index, query_limit);
  }

  return 0;
}
