CXX = g++
CXXFLAGS = -O3 -std=c++20 -march=x86-64 -msse4.2 -Wall -Wextra -pthread -MMD -MP -fopenmp

ORAM_DIR     := include/H2O2RAM
ORAM_INCLUDE := -I$(ORAM_DIR)/include
ORAM_SRC     := $(ORAM_DIR)/src/depthCounter.cpp \
                $(ORAM_DIR)/src/oblivious_operations.cpp \
                $(ORAM_DIR)/src/prp.cpp \
                $(ORAM_DIR)/src/timer.cpp

CXXFLAGS += $(ORAM_INCLUDE)

SRCS = src/main.cpp src/index/hnsw.cpp src/oblivious_index/hnsw.cpp $(ORAM_SRC)
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)
TARGET = hnsw_eval

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -ltbb -lcrypto

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
