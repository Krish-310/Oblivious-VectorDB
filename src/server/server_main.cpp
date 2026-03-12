// =============================================================================
// server_main.cpp
//
// Oblivious VectorDB Block Server
//
// A dumb encrypted block store that the ORAM client talks to over TCP.
// The server never decrypts anything — it stores and retrieves opaque bytes.
//
// Usage:
//   ./oram_server [--port <port>] [--data-dir <path>]
//
// Defaults: port=7777, data-dir=./block_store
// =============================================================================

#include "../shared/protocol.h"
#include "block_store.h"

#include <arpa/inet.h>
#include <csignal>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static volatile bool g_running = true;
static void sig_handler(int) { g_running = false; }

// Handle one client connection until it closes
void handle_client(int client_fd, server::BlockStore &store) {
  std::cout << "[server] Client connected fd=" << client_fd << "\n";
  try {
    while (true) {
      std::vector<uint8_t> payload;
      uint8_t op = proto::recv_msg(client_fd, payload);

      switch (op) {
      case proto::OP_READ_BLOCK: {
        uint32_t level;
        uint64_t slot;
        proto::parse_rw_block_header(payload, level, slot);
        auto block = store.read_block(level, slot);
        auto resp =
            proto::make_response(proto::STATUS_OK, block.data(), block.size());
        proto::send_msg(client_fd, proto::OP_RESPONSE, resp);
        std::cout << "[server] READ_BLOCK level=" << level << " slot=" << slot
                  << " size=" << block.size() << "B\n";
        break;
      }
      case proto::OP_WRITE_BLOCK: {
        uint32_t level;
        uint64_t slot;
        proto::parse_rw_block_header(payload, level, slot);
        std::vector<uint8_t> block(payload.begin() + 12, payload.end());
        store.write_block(level, slot, block);
        auto resp = proto::make_response(proto::STATUS_OK);
        proto::send_msg(client_fd, proto::OP_RESPONSE, resp);
        std::cout << "[server] WRITE_BLOCK level=" << level << " slot=" << slot
                  << "\n";
        break;
      }
      case proto::OP_READ_LEVEL: {
        uint32_t level = proto::parse_level(payload);
        auto data = store.read_level(level);
        auto resp =
            proto::make_response(proto::STATUS_OK, data.data(), data.size());
        proto::send_msg(client_fd, proto::OP_RESPONSE, resp);
        std::cout << "[server] READ_LEVEL level=" << level
                  << " size=" << data.size() << "B\n";
        break;
      }
      case proto::OP_WRITE_LEVEL: {
        uint32_t level = proto::parse_level(payload);
        // Remaining bytes after the 4B level header are the level payload
        std::vector<uint8_t> level_data(payload.begin() + 4, payload.end());
        store.write_level(level, level_data);
        auto resp = proto::make_response(proto::STATUS_OK);
        proto::send_msg(client_fd, proto::OP_RESPONSE, resp);
        std::cout << "[server] WRITE_LEVEL level=" << level
                  << " size=" << level_data.size() << "B\n";
        break;
      }
      default:
        std::cerr << "[server] Unknown opcode 0x" << std::hex << (int)op
                  << std::dec << " — closing.\n";
        close(client_fd);
        return;
      }
    }
  } catch (const std::exception &e) {
    // Clean disconnection or error
    std::cout << "[server] Client fd=" << client_fd
              << " disconnected: " << e.what() << "\n";
  }
  close(client_fd);
}

int main(int argc, char **argv) {
  uint16_t port = 7777;
  std::string data_dir = "./block_store";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc)
      port = static_cast<uint16_t>(std::stoi(argv[++i]));
    if (a == "--data-dir" && i + 1 < argc)
      data_dir = argv[++i];
  }

  signal(SIGINT, sig_handler);
  signal(SIGTERM, sig_handler);

  server::BlockStore store(data_dir);
  std::cout << "[server] Starting on port " << port << " data-dir=" << data_dir
            << "\n";

  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    perror("socket");
    return 1;
  }

  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    perror("bind");
    return 1;
  }
  if (listen(listen_fd, 1) < 0) {
    perror("listen");
    return 1;
  }

  std::cout << "[server] Listening for connections...\n";

  while (g_running) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(
        listen_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (g_running)
        perror("accept");
      break;
    }
    // Single-threaded: handle one client at a time
    handle_client(client_fd, store);
  }

  close(listen_fd);
  std::cout << "[server] Shutdown.\n";
  return 0;
}
