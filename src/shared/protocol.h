#pragma once
// =============================================================================
// protocol.h
//
// Compact binary protocol for the Oblivious VectorDB client-server RPC layer.
// All messages are framed as:
//
//   [1 byte opcode][4 bytes payload_len (big-endian)][payload_len bytes
//   payload]
//
// The server responds to every request with a RESPONSE message:
//
//   [0xFF][4 bytes len][4 bytes status (0=ok, 1=err)][data ...]
//
// Both client and server use the send_msg / recv_msg helpers below.
// =============================================================================

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace proto {

// Opcodes
constexpr uint8_t OP_READ_BLOCK = 0x01; // payload: [4B level][8B slot]
constexpr uint8_t OP_WRITE_BLOCK =
    0x02; // payload: [4B level][8B slot][N bytes block]
constexpr uint8_t OP_READ_LEVEL = 0x03;  // payload: [4B level]
constexpr uint8_t OP_WRITE_LEVEL = 0x04; // payload: [4B level][N bytes data]
constexpr uint8_t OP_RESPONSE = 0xFF;

constexpr uint32_t STATUS_OK = 0;
constexpr uint32_t STATUS_ERR = 1;

// ---------------------------------------------------------------------------
// Low-level I/O helpers — guaranteed full reads/writes on a stream socket
// ---------------------------------------------------------------------------
inline void write_all(int fd, const void *buf, size_t n) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0)
      throw std::runtime_error("proto::write_all: connection closed");
    p += w;
    n -= w;
  }
}

inline void read_all(int fd, void *buf, size_t n) {
  uint8_t *p = static_cast<uint8_t *>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0)
      throw std::runtime_error("proto::read_all: connection closed");
    p += r;
    n -= r;
  }
}

// ---------------------------------------------------------------------------
// Framing: send a message with a given opcode and payload bytes
// ---------------------------------------------------------------------------
inline void send_msg(int fd, uint8_t opcode,
                     const std::vector<uint8_t> &payload) {
  uint32_t len_be = htonl(static_cast<uint32_t>(payload.size()));
  write_all(fd, &opcode, 1);
  write_all(fd, &len_be, 4);
  if (!payload.empty())
    write_all(fd, payload.data(), payload.size());
}

// Receive a message: returns opcode and fills payload
inline uint8_t recv_msg(int fd, std::vector<uint8_t> &payload_out) {
  uint8_t opcode;
  uint32_t len_be;
  read_all(fd, &opcode, 1);
  read_all(fd, &len_be, 4);
  uint32_t len = ntohl(len_be);
  payload_out.resize(len);
  if (len > 0)
    read_all(fd, payload_out.data(), len);
  return opcode;
}

// ---------------------------------------------------------------------------
// Convenience: build payloads
// ---------------------------------------------------------------------------
inline std::vector<uint8_t> make_read_block(uint32_t level, uint64_t slot) {
  std::vector<uint8_t> p(12);
  uint32_t l_be = htonl(level);
  uint64_t s_be = htobe64(slot);
  memcpy(p.data(), &l_be, 4);
  memcpy(p.data() + 4, &s_be, 8);
  return p;
}

inline std::vector<uint8_t> make_write_block(uint32_t level, uint64_t slot,
                                             const uint8_t *block_data,
                                             size_t block_size) {
  std::vector<uint8_t> p(12 + block_size);
  uint32_t l_be = htonl(level);
  uint64_t s_be = htobe64(slot);
  memcpy(p.data(), &l_be, 4);
  memcpy(p.data() + 4, &s_be, 8);
  memcpy(p.data() + 12, block_data, block_size);
  return p;
}

inline std::vector<uint8_t> make_read_level(uint32_t level) {
  std::vector<uint8_t> p(4);
  uint32_t l_be = htonl(level);
  memcpy(p.data(), &l_be, 4);
  return p;
}

inline std::vector<uint8_t> make_write_level(uint32_t level,
                                             const uint8_t *data, size_t sz) {
  std::vector<uint8_t> p(4 + sz);
  uint32_t l_be = htonl(level);
  memcpy(p.data(), &l_be, 4);
  memcpy(p.data() + 4, data, sz);
  return p;
}

inline std::vector<uint8_t>
make_response(uint32_t status, const uint8_t *data = nullptr, size_t sz = 0) {
  std::vector<uint8_t> p(4 + sz);
  uint32_t s_be = htonl(status);
  memcpy(p.data(), &s_be, 4);
  if (data && sz)
    memcpy(p.data() + 4, data, sz);
  return p;
}

// ---------------------------------------------------------------------------
// Parse helpers (extract fields from raw payload buffers)
// ---------------------------------------------------------------------------
inline void parse_rw_block_header(const std::vector<uint8_t> &p,
                                  uint32_t &level, uint64_t &slot) {
  uint32_t l_be;
  uint64_t s_be;
  memcpy(&l_be, p.data(), 4);
  memcpy(&s_be, p.data() + 4, 8);
  level = ntohl(l_be);
  slot = be64toh(s_be);
}

inline uint32_t parse_level(const std::vector<uint8_t> &p) {
  uint32_t l_be;
  memcpy(&l_be, p.data(), 4);
  return ntohl(l_be);
}

inline uint32_t parse_response_status(const std::vector<uint8_t> &p) {
  uint32_t s_be;
  memcpy(&s_be, p.data(), 4);
  return ntohl(s_be);
}

} // namespace proto
