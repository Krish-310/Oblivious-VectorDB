CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -Wextra -pthread -MMD -MP

ORAM_DIR     := third_party/H2O2RAM
ORAM_INCLUDE := -I$(ORAM_DIR)/include
ORAM_SRC     := $(ORAM_DIR)/src/omap.cpp \
                $(ORAM_DIR)/src/oram.cpp \
                $(ORAM_DIR)/src/hash_table.cpp

CXXFLAGS += $(ORAM_INCLUDE)

SRCS = src/main.cpp src/index/hnsw.cpp $(ORAM_SRC)
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)
TARGET = hnsw_eval

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
