CXX      = g++
CXXFLAGS = -O3 -std=c++20 -march=x86-64 -msse4.2 -Wall -Wextra -pthread -MMD -MP -fopenmp

ORAM_DIR     := third_party/H2O2RAM
ORAM_INCLUDE := -I$(ORAM_DIR)/include
ORAM_SRC     := $(ORAM_DIR)/src/depthCounter.cpp \
                $(ORAM_DIR)/src/oblivious_operations.cpp \
                $(ORAM_DIR)/src/prp.cpp \
                $(ORAM_DIR)/src/timer.cpp

CXXFLAGS += $(ORAM_INCLUDE) -Isrc

# =============================================================================
# Client (hnsw_eval) — HNSW + all storage backends + H2O2RAM
# =============================================================================
CLIENT_SRCS = src/main.cpp src/index/hnsw.cpp $(ORAM_SRC)
CLIENT_OBJS = $(CLIENT_SRCS:.cpp=.o)
CLIENT_DEPS = $(CLIENT_SRCS:.cpp=.d)
CLIENT_TARGET = hnsw_eval

# =============================================================================
# Server (oram_server) — dumb block store, no H2O2RAM dependency
# =============================================================================
SERVER_SRCS = src/server/server_main.cpp
SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)
SERVER_DEPS = $(SERVER_SRCS:.cpp=.d)
SERVER_TARGET = oram_server

LDFLAGS_CLIENT = -ltbb -lcrypto
LDFLAGS_SERVER =  # no extra libs needed for the server

all: $(CLIENT_TARGET) $(SERVER_TARGET)

client: $(CLIENT_TARGET)
server: $(SERVER_TARGET)

$(CLIENT_TARGET): $(CLIENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_CLIENT)

$(SERVER_TARGET): $(SERVER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_SERVER)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(CLIENT_DEPS) $(SERVER_DEPS)

clean:
	rm -f $(CLIENT_OBJS) $(CLIENT_DEPS) $(CLIENT_TARGET)
	rm -f $(SERVER_OBJS) $(SERVER_DEPS) $(SERVER_TARGET)

.PHONY: all client server clean
