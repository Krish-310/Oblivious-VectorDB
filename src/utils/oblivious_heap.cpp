// oblivious heap implementations.
#include "oblivious_heap.h"
#include <limits>

namespace utils {

ObliviousMinHeap::ObliviousMinHeap() {}
ObliviousMinHeap::~ObliviousMinHeap() {}

void ObliviousMinHeap::insert(float dist, int id) {
    data_.push_back({dist, id});
}

std::pair<float, int> ObliviousMinHeap::pop_min() {
    if (data_.empty()) {
        return {std::numeric_limits<float>::infinity(), -1};
    }

    float min_dist = std::numeric_limits<float>::infinity();
    int min_id = -1;
    size_t min_idx = 0;

    // Linear scan over ALL elements to find the minimum distance
    for (size_t i = 0; i < data_.size(); ++i) {
        float current_dist = data_[i].first;
        if (current_dist < min_dist) {
            min_dist = current_dist;
            min_id = data_[i].second;
            min_idx = i;
        }
    }

    // Oblivious removal: replace the minimum element with the last element.
    // To hide the index being updated, we linearly scan all elements and perform the update restrictively.
    std::pair<float, int> last_element = data_.back();
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i == min_idx) {
            data_[i] = last_element;
        }
    }
    
    // Remove the last element since it was moved to the min_idx position
    data_.pop_back();

    return {min_dist, min_id};
}

std::pair<float, int> ObliviousMinHeap::get_min() const {
    if (data_.empty()) {
        return {std::numeric_limits<float>::infinity(), -1};
    }

    float min_dist = std::numeric_limits<float>::infinity();
    int min_id = -1;

    // Linear scan to return min without mutating
    for (size_t i = 0; i < data_.size(); ++i) {
        float current_dist = data_[i].first;
        if (current_dist < min_dist) {
            min_dist = current_dist;
            min_id = data_[i].second;
        }
    }

    return {min_dist, min_id};
}

bool ObliviousMinHeap::empty() const {
    return data_.empty();
}

size_t ObliviousMinHeap::size() const {
    return data_.size();
}

ObliviousMaxHeap::ObliviousMaxHeap() {}
ObliviousMaxHeap::~ObliviousMaxHeap() {}

void ObliviousMaxHeap::insert(float dist, int id) {
    data_.push_back({dist, id});
}

std::pair<float, int> ObliviousMaxHeap::pop_max() {
    if (data_.empty()) {
        return {-std::numeric_limits<float>::infinity(), -1};
    }

    float max_dist = -std::numeric_limits<float>::infinity();
    int max_id = -1;
    size_t max_idx = 0;

    // Linear scan to find the maximum
    for (size_t i = 0; i < data_.size(); ++i) {
        float current_dist = data_[i].first;
        if (current_dist > max_dist) {
            max_dist = current_dist;
            max_id = data_[i].second;
            max_idx = i;
        }
    }

    // Oblivious removal
    std::pair<float, int> last_element = data_.back();
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i == max_idx) {
            data_[i] = last_element;
        }
    }
    
    data_.pop_back();

    return {max_dist, max_id};
}

std::pair<float, int> ObliviousMaxHeap::get_max() const {
    if (data_.empty()) {
        return {-std::numeric_limits<float>::infinity(), -1};
    }

    float max_dist = -std::numeric_limits<float>::infinity();
    int max_id = -1;

    for (size_t i = 0; i < data_.size(); ++i) {
        float current_dist = data_[i].first;
        if (current_dist > max_dist) {
            max_dist = current_dist;
            max_id = data_[i].second;
        }
    }

    return {max_dist, max_id};
}

bool ObliviousMaxHeap::empty() const {
    return data_.empty();
}

size_t ObliviousMaxHeap::size() const {
    return data_.size();
}

} // namespace utils