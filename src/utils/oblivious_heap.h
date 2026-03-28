#ifndef OBLIVIOUS_HEAP_H
#define OBLIVIOUS_HEAP_H

// Minheap that is implemented as a vector
#include <vector>
#include <utility>
#include <cstddef>

// can only be used for dist_pair

namespace utils {

class ObliviousMinHeap {
public:
    ObliviousMinHeap();
    ~ObliviousMinHeap();

    void insert(float dist, int id);
    std::pair<float, int> pop_min();
    std::pair<float, int> get_min() const;
    
    bool empty() const;
    size_t size() const;

private:
    std::vector<std::pair<float, int>> data_;
};

class ObliviousMaxHeap {
public:
    ObliviousMaxHeap();
    ~ObliviousMaxHeap();

    void insert(float dist, int id);
    std::pair<float, int> pop_max();
    std::pair<float, int> get_max() const;
    
    bool empty() const;
    size_t size() const;

private:
    std::vector<std::pair<float, int>> data_;
};

} // namespace utils

#endif // OBLIVIOUS_HEAP_H