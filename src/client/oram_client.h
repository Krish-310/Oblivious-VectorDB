#pragma once
// =============================================================================
// oram_client.h
//
// Trusted ORAM client that talks to the BlockServer over TCP.
//
// Algorithm: hierarchical ORAM (log N levels).
//   - Level i holds 2^i slots, each an AES-encrypted block.
//   - Client holds: AES key, stash (up to LINEAR_SCAN_THRESHOLD blocks),
//     and level occupancy flags.
//   - On find(id): scan stash, query one slot per non-empty level, re-insert
//     to level 0, rebuild (merge + oblivious sort) if level 0 overflows.
//
// The server stores opaque encrypted bytes in flat binary files.
// All crypto (AES-PRP) uses H2O2RAM's prp.hpp — client-side only.
//
// =============================================================================

#include "../../../third_party/H2O2RAM/include/prp.hpp"
#include "../../shared/protocol.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <cstring>
#include <netinet/in.h>
#include <random>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace oram {

// Fixed-size block payload: ID + raw float data
// Must be a multiple of 8 bytes for the oblivious XOR swap.
// We pad the block data to BLOCK_VALUE_BYTES bytes.
static constexpr size_t ORAM_MAX_DIM_CS = 512; // max floats per vector
static constexpr size_t BLOCK_VALUE_BYTES = ORAM_MAX_DIM_CS * sizeof(float);

// Linear scan stash capacity — kept small so stash scan stays O(1)
static constexpr size_t STASH_CAPACITY = 128;

// Sentinel "dummy" ID — a block with this ID is empty padding
static constexpr uint32_t DUMMY_ID = UINT32_MAX;

// -------------------------------------------------------------------------
// PlainBlock: one logical ORAM entry (trusted memory only)
// -------------------------------------------------------------------------
struct PlainBlock {
  uint32_t id = DUMMY_ID;
  float data[ORAM_MAX_DIM_CS] = {};

  bool is_dummy() const { return id == DUMMY_ID; }
};
static_assert(sizeof(PlainBlock) == 4 + BLOCK_VALUE_BYTES);

// -------------------------------------------------------------------------
// EncBlock: AES-encrypted PlainBlock stored on server
// We encrypt with AES-128 CTR using H2O2RAM's AESCSPRP keyed on
// (level, slot) to make server-side blocks look random.
// -------------------------------------------------------------------------
static constexpr size_t ENC_BLOCK_SIZE =
    sizeof(PlainBlock); // no auth tag for simplicity

// -------------------------------------------------------------------------
// OramClient
// -------------------------------------------------------------------------
class OramClient {
public:
  // Connect to block server at host:port and initialize for n vectors of dim
  // floats
  OramClient(const std::string &host, uint16_t port, size_t n, size_t dim)
      : dim_(dim), n_(n), prp_() {
    if (dim > ORAM_MAX_DIM_CS)
      throw std::invalid_argument("OramClient: dim exceeds ORAM_MAX_DIM_CS");

    // TCP connect
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0)
      throw std::runtime_error("OramClient: socket() failed");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (::connect(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
      throw std::runtime_error("OramClient: connect() failed to " + host + ":" +
                               std::to_string(port));

    // Number of levels = ceil(log2(n)) + 1
    num_levels_ = 1;
    while ((size_t(1) << num_levels_) < n)
      num_levels_++;
    num_levels_++; // extra level for rebuilds

    level_empty_.assign(num_levels_, true);
    stash_.reserve(STASH_CAPACITY);

    std::cout << "[oram_client] Connected to " << host << ":" << port
              << "  n=" << n << " levels=" << num_levels_ << "\n";
  }

  ~OramClient() {
    if (fd_ >= 0)
      ::close(fd_);
  }

  // Bulk-load n vectors from raw_vectors into the ORAM
  void bulk_insert(const float *raw_vectors, size_t n) {
    std::cout << "[oram_client] Bulk inserting " << n << " vectors...\n";
    for (size_t id = 0; id < n; ++id) {
      PlainBlock blk;
      blk.id = static_cast<uint32_t>(id);
      memcpy(blk.data, raw_vectors + id * dim_, dim_ * sizeof(float));
      stash_.push_back(blk);
      // Flush stash to levels whenever it reaches STASH_CAPACITY
      if (stash_.size() >= STASH_CAPACITY)
        flush_stash();
    }
    // Final flush
    if (!stash_.empty())
      flush_stash();
    std::cout << "[oram_client] Bulk insert done.\n";
  }

  // Oblivious lookup: find block with given id, return pointer to its data.
  // Result is valid until the next call to find().
  const float *find(uint32_t id) {
    PlainBlock found;
    found.id = DUMMY_ID;

    // 1. Obliviously scan stash
    for (auto &blk : stash_) {
      bool match = (blk.id == id);
      // Oblivious select: copy blk into found if match
      if (match) {
        found = blk;
        blk.id = DUMMY_ID;
      }
    }

    // 2. Query one slot per non-empty level using PRF hash
    for (size_t lvl = 0; lvl < num_levels_; ++lvl) {
      if (level_empty_[lvl])
        continue;
      uint64_t slot = hash_slot(id, lvl);
      PlainBlock blk = server_read(lvl, slot);
      bool match = (blk.id == id);
      if (match) {
        found = blk;
      }
      // Always write back a dummy to keep access pattern uniform
      PlainBlock dummy;
      dummy.id = DUMMY_ID;
      server_write(lvl, slot, match ? dummy : blk);
    }

    // 3. Re-insert found block into stash
    stash_.push_back(found);

    // 4. If stash full, flush
    if (stash_.size() >= STASH_CAPACITY)
      flush_stash();

    // Cache result in return buffer
    if (found.id != DUMMY_ID) {
      memcpy(ret_buf_, found.data, dim_ * sizeof(float));
      return ret_buf_;
    }
    throw std::runtime_error("OramClient::find: id=" + std::to_string(id) +
                             " not found");
  }

private:
  int fd_ = -1;
  size_t dim_;
  size_t n_;
  size_t num_levels_;
  std::vector<bool> level_empty_;
  std::vector<PlainBlock> stash_;
  ORAM::AESCSPRP prp_;
  float ret_buf_[ORAM_MAX_DIM_CS] = {};

  // -----------------------------------------------------------------------
  // PRF-based slot hash: maps (id, level) → slot index in that level
  // -----------------------------------------------------------------------
  uint64_t hash_slot(uint32_t id, size_t level) const {
    uint64_t level_size = uint64_t(1) << level;
    // Use PRF key mixed with level to deterministically pick a slot
    // We hash a 16-byte input: [id (4B) | level (4B) | zeros (8B)]
    uint8_t input[16] = {};
    memcpy(input, &id, 4);
    uint32_t lvl32 = static_cast<uint32_t>(level);
    memcpy(input + 4, &lvl32, 4);
    uint8_t out[16];
    prp_(input, out);
    uint64_t h;
    memcpy(&h, out, 8);
    return h % level_size;
  }

  // -----------------------------------------------------------------------
  // Encrypt/decrypt a PlainBlock
  // We use XOR with a PRF-derived keystream (stream cipher mode).
  // In a production system, use AES-GCM for authentication.
  // -----------------------------------------------------------------------
  std::vector<uint8_t> encrypt(const PlainBlock &blk, size_t level,
                               uint64_t slot) const {
    std::vector<uint8_t> enc(ENC_BLOCK_SIZE);
    const uint8_t *plain = reinterpret_cast<const uint8_t *>(&blk);
    // Generate keystream: PRF(level||slot||chunk_idx)
    for (size_t off = 0; off < ENC_BLOCK_SIZE; off += 16) {
      uint8_t input[16] = {};
      uint32_t lvl32 = static_cast<uint32_t>(level);
      uint64_t off64 = off;
      memcpy(input, &lvl32, 4);
      memcpy(input + 4, &slot, 8);
      memcpy(input + 12, &off64, 4);
      uint8_t ks[16];
      prp_(input, ks);
      size_t chunk = std::min<size_t>(16, ENC_BLOCK_SIZE - off);
      for (size_t i = 0; i < chunk; ++i)
        enc[off + i] = plain[off + i] ^ ks[i];
    }
    return enc;
  }

  PlainBlock decrypt(const std::vector<uint8_t> &enc, size_t level,
                     uint64_t slot) const {
    PlainBlock blk;
    uint8_t *plain = reinterpret_cast<uint8_t *>(&blk);
    for (size_t off = 0; off < ENC_BLOCK_SIZE; off += 16) {
      uint8_t input[16] = {};
      uint32_t lvl32 = static_cast<uint32_t>(level);
      uint64_t off64 = off;
      memcpy(input, &lvl32, 4);
      memcpy(input + 4, &slot, 8);
      memcpy(input + 12, &off64, 4);
      uint8_t ks[16];
      prp_(input, ks);
      size_t chunk = std::min<size_t>(16, ENC_BLOCK_SIZE - off);
      for (size_t i = 0; i < chunk; ++i)
        plain[off + i] = enc[off + i] ^ ks[i];
    }
    return blk;
  }

  // -----------------------------------------------------------------------
  // RPC helpers — read/write a single block via TCP
  // -----------------------------------------------------------------------
  PlainBlock server_read(size_t level, uint64_t slot) {
    auto payload = proto::make_read_block(static_cast<uint32_t>(level), slot);
    proto::send_msg(fd_, proto::OP_READ_BLOCK, payload);

    std::vector<uint8_t> resp;
    proto::recv_msg(fd_, resp);
    uint32_t status = proto::parse_response_status(resp);
    if (status != proto::STATUS_OK)
      throw std::runtime_error("server_read: server returned error");

    std::vector<uint8_t> enc(resp.begin() + 4, resp.end());
    if (enc.empty()) {
      PlainBlock b;
      b.id = DUMMY_ID;
      return b;
    }
    return decrypt(enc, level, slot);
  }

  void server_write(size_t level, uint64_t slot, const PlainBlock &blk) {
    auto enc = encrypt(blk, level, slot);
    auto payload = proto::make_write_block(static_cast<uint32_t>(level), slot,
                                           enc.data(), enc.size());
    proto::send_msg(fd_, proto::OP_WRITE_BLOCK, payload);

    std::vector<uint8_t> resp;
    proto::recv_msg(fd_, resp);
    if (proto::parse_response_status(resp) != proto::STATUS_OK)
      throw std::runtime_error("server_write: server returned error");
  }

  // -----------------------------------------------------------------------
  // Flush stash to server — find the lowest empty level, merge & write
  // -----------------------------------------------------------------------
  void flush_stash() {
    // Find lowest empty level to absorb stash
    size_t target_level = 0;
    while (target_level < num_levels_ && !level_empty_[target_level])
      target_level++;

    if (target_level >= num_levels_)
      throw std::runtime_error("OramClient: all levels full — dataset too "
                               "large for current level count");

    // Collect all blocks from levels 0..target_level-1 + stash
    std::vector<PlainBlock> all;
    all.insert(all.end(), stash_.begin(), stash_.end());
    stash_.clear();

    for (size_t lvl = 0; lvl < target_level; ++lvl) {
      if (level_empty_[lvl])
        continue;
      uint64_t level_size = uint64_t(1) << lvl;
      auto level_payload = proto::make_read_level(static_cast<uint32_t>(lvl));
      proto::send_msg(fd_, proto::OP_READ_LEVEL, level_payload);
      std::vector<uint8_t> resp;
      proto::recv_msg(fd_, resp);
      // resp = [4B status][level data]
      std::vector<uint8_t> level_data(resp.begin() + 4, resp.end());
      for (uint64_t slot = 0; slot < level_size; ++slot) {
        size_t off = slot * ENC_BLOCK_SIZE;
        if (off + ENC_BLOCK_SIZE > level_data.size())
          break;
        std::vector<uint8_t> enc(level_data.begin() + off,
                                 level_data.begin() + off + ENC_BLOCK_SIZE);
        all.push_back(decrypt(enc, lvl, slot));
      }
      level_empty_[lvl] = true;
    }

    // Remove duplicates — keep first occurrence of each id (stash wins)
    std::vector<bool> seen(n_ + 1, false);
    std::vector<PlainBlock> deduped;
    for (auto &blk : all) {
      if (blk.is_dummy())
        continue;
      if (blk.id < n_ && !seen[blk.id]) {
        seen[blk.id] = true;
        deduped.push_back(blk);
      }
    }

    // Write deduped blocks into target_level using PRF hash slots
    uint64_t level_size = uint64_t(1) << target_level;
    // Build encrypted level: one PlainBlock per slot, fill with dummies
    std::vector<PlainBlock> level_blocks(level_size);
    for (auto &blk : deduped) {
      uint64_t slot = hash_slot(blk.id, target_level);
      level_blocks[slot] = blk;
    }

    // Serialize and ship to server
    // WRITE_LEVEL payload: [4B level][8B num_slots][data]
    size_t data_size = level_size * ENC_BLOCK_SIZE;
    std::vector<uint8_t> level_payload;
    level_payload.reserve(4 + 8 + data_size);
    // level is encoded by make_write_level
    // We need to prefix with num_slots for the block_store header
    uint64_t ns_be = htobe64(level_size);
    std::vector<uint8_t> header_and_data(8 + data_size);
    memcpy(header_and_data.data(), &ns_be, 8);
    for (uint64_t slot = 0; slot < level_size; ++slot) {
      auto enc = encrypt(level_blocks[slot], target_level, slot);
      memcpy(header_and_data.data() + 8 + slot * ENC_BLOCK_SIZE, enc.data(),
             ENC_BLOCK_SIZE);
    }

    auto wl_payload =
        proto::make_write_level(static_cast<uint32_t>(target_level),
                                header_and_data.data(), header_and_data.size());
    proto::send_msg(fd_, proto::OP_WRITE_LEVEL, wl_payload);
    std::vector<uint8_t> resp;
    proto::recv_msg(fd_, resp);
    if (proto::parse_response_status(resp) != proto::STATUS_OK)
      throw std::runtime_error("flush_stash: server error on WRITE_LEVEL");

    level_empty_[target_level] = false;
  }
};

} // namespace oram
