#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnsw {

template <typename T>
void read_binary_matrix(const std::string &filename, std::vector<T> &data,
                        int &num_points, int &dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  in.read(reinterpret_cast<char *>(&num_points), sizeof(int));
  in.read(reinterpret_cast<char *>(&dim), sizeof(int));

  std::cout << "Loading " << filename << " with shape (" << num_points << ", "
            << dim << ")\n";

  data.resize(num_points * dim);
  in.read(reinterpret_cast<char *>(data.data()), num_points * dim * sizeof(T));
  in.close();
}

class StopW {
  std::chrono::steady_clock::time_point time_begin;

public:
  StopW() { time_begin = std::chrono::steady_clock::now(); }
  float getElapsedTimeMicro() {
    std::chrono::steady_clock::time_point time_end =
        std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(time_end -
                                                                  time_begin)
                .count());
  }
  void reset() { time_begin = std::chrono::steady_clock::now(); }
};

} // namespace hnsw

#endif // UTILS_H
