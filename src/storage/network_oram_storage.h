#pragma once
// =============================================================================
// network_oram_storage.h
//
// A StorageAdapter that stores vectors obliviously on a remote BlockServer.
// Uses OramClient to make encrypted RPC calls — the server sees only
// ciphertext.
//
// Drop-in replacement for OramStorage (local) or DiskStorage:
//
//   auto storage = std::make_unique<NetworkOramStorage>(
//       "127.0.0.1", 7777, raw_vectors, base_n, base_dim);
//   index.set_storage(storage.get());
// =============================================================================

#include "../client/oram_client.h"
#include "storage_adapter.h"

#include <memory>
#include <stdexcept>

namespace hnsw {
namespace storage {

class NetworkOramStorage : public StorageAdapter {
public:
  NetworkOramStorage(const std::string &host, uint16_t port,
                     const float *raw_vectors, size_t n, size_t dim)
      : dim_(dim), n_(n),
        client_(std::make_unique<oram::OramClient>(host, port, n, dim)) {
    std::cout << "[NetworkOramStorage] Bulk-loading " << n
              << " vectors into remote ORAM...\n";
    client_->bulk_insert(raw_vectors, n);
    std::cout << "[NetworkOramStorage] Ready.\n";
  }

  const float *get_vector(size_t id) override {
    if (id >= n_)
      throw std::out_of_range(
          "NetworkOramStorage::get_vector: id out of range");
    return client_->find(static_cast<uint32_t>(id));
  }

  size_t get_dim() const override { return dim_; }
  size_t size() const { return n_; }

private:
  size_t dim_;
  size_t n_;
  std::unique_ptr<oram::OramClient> client_;
};

} // namespace storage
} // namespace hnsw
