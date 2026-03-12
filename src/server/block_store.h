#pragma once
// =============================================================================
// block_store.h
//
// Server-side encrypted block storage. Manages one binary file per ORAM level.
// Files are flat arrays of fixed-size blocks; the server stores them opaquely —
// it never decrypts, never interprets.
//
// Layout of level_NN.bin:
//   [8B num_slots][block_size * num_slots bytes of raw encrypted data]
//
// The server does not know block_size at startup — the client communicates it
// as part of the first WRITE_LEVEL or WRITE_BLOCK call (via the payload size).
// After the first write the server infers and caches block_size.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace server {

class BlockStore {
public:
  explicit BlockStore(std::string data_dir) : data_dir_(std::move(data_dir)) {
    std::filesystem::create_directories(data_dir_);
  }

  // -------------------------------------------------------------------------
  // read_block: return the raw bytes of slot `slot` in level `level`.
  // Returns an empty vector if the level file does not exist.
  // -------------------------------------------------------------------------
  std::vector<uint8_t> read_block(uint32_t level, uint64_t slot) const {
    auto path = level_path(level);
    if (!std::filesystem::exists(path))
      return {};

    auto [block_size, num_slots] = read_header(path);
    if (slot >= num_slots)
      throw std::out_of_range("BlockStore::read_block: slot out of range");

    std::ifstream f(path, std::ios::binary);
    f.seekg(8 + slot * block_size);
    std::vector<uint8_t> buf(block_size);
    f.read(reinterpret_cast<char *>(buf.data()), block_size);
    return buf;
  }

  // -------------------------------------------------------------------------
  // write_block: overwrite a single slot. Level file must already exist.
  // -------------------------------------------------------------------------
  void write_block(uint32_t level, uint64_t slot,
                   const std::vector<uint8_t> &data) {
    auto path = level_path(level);
    if (!std::filesystem::exists(path))
      throw std::runtime_error("BlockStore::write_block: level file does not "
                               "exist; call write_level first");

    auto [block_size, num_slots] = read_header(path);
    if (data.size() != block_size)
      throw std::invalid_argument(
          "BlockStore::write_block: block size mismatch");
    if (slot >= num_slots)
      throw std::out_of_range("BlockStore::write_block: slot out of range");

    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(8 + slot * block_size);
    f.write(reinterpret_cast<const char *>(data.data()), block_size);
  }

  // -------------------------------------------------------------------------
  // read_level: return the entire level file's block array (without header).
  // -------------------------------------------------------------------------
  std::vector<uint8_t> read_level(uint32_t level) const {
    auto path = level_path(level);
    if (!std::filesystem::exists(path))
      return {};

    auto [block_size, num_slots] = read_header(path);
    size_t data_size = block_size * num_slots;

    std::ifstream f(path, std::ios::binary);
    f.seekg(8);
    std::vector<uint8_t> buf(data_size);
    f.read(reinterpret_cast<char *>(buf.data()), data_size);
    return buf;
  }

  // -------------------------------------------------------------------------
  // write_level: create or overwrite the level file entirely.
  //   data.size() must be divisible by block_size (inferred if level exists,
  //   otherwise set on first write — client sends num_slots in header bytes).
  //
  // Protocol: client sends [8B num_slots][num_slots * block_size bytes]
  // -------------------------------------------------------------------------
  void write_level(uint32_t level, const std::vector<uint8_t> &payload) {
    // First 8 bytes of payload = num_slots (big-endian uint64)
    if (payload.size() < 8)
      throw std::invalid_argument("BlockStore::write_level: payload too short");
    uint64_t num_slots_be;
    memcpy(&num_slots_be, payload.data(), 8);
    uint64_t num_slots = be64toh(num_slots_be);

    size_t data_size = payload.size() - 8;
    size_t block_size = (num_slots > 0) ? (data_size / num_slots) : 0;

    auto path = level_path(level);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    // Write header: [8B num_slots] encoded as two uint32s for portability
    // We reuse the same 8-byte num_slots field
    f.write(reinterpret_cast<const char *>(&num_slots_be), 8);
    // Write payload data (excluding the 8-byte num_slots header)
    if (data_size > 0)
      f.write(reinterpret_cast<const char *>(payload.data() + 8), data_size);

    // Cache block size for this level
    block_sizes_[level] = block_size;
    (void)num_slots; // already used above
  }

  size_t num_levels() const {
    size_t count = 0;
    for (auto &e : std::filesystem::directory_iterator(data_dir_))
      if (e.path().extension() == ".bin")
        count++;
    return count;
  }

private:
  std::string data_dir_;
  mutable std::unordered_map<uint32_t, size_t> block_sizes_;

  std::string level_path(uint32_t level) const {
    char buf[64];
    snprintf(buf, sizeof(buf), "/level_%02u.bin", level);
    return data_dir_ + buf;
  }

  // Read 8-byte header: returns {block_size, num_slots}
  std::pair<size_t, uint64_t> read_header(const std::string &path) const {
    std::ifstream f(path, std::ios::binary);
    uint64_t num_slots_be;
    f.read(reinterpret_cast<char *>(&num_slots_be), 8);
    uint64_t num_slots = be64toh(num_slots_be);

    size_t file_size = std::filesystem::file_size(path);
    size_t data_size = file_size - 8;
    size_t block_size = (num_slots > 0) ? (data_size / num_slots) : 0;
    return {block_size, num_slots};
  }
};

} // namespace server
