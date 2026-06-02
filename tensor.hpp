#include <vector>
#include <iostream>
#include <random>
#include <cmath>

struct Tensor {
    size_t rows;
    size_t cols;
    std::vector<float> data;

    Tensor(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {};

    inline float& at(size_t r, size_t c) {
        return data[r * cols + c];
    }

    inline float at(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    // 1. For Dummy Sensor Data and Targets
    void randomize_uniform() {
        std::random_device rd; // Hardware entropy source
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dis(gen);
        }
    }

    // 2. For Learnable Weight Matrices (W_fuse, W_q, W_k, W_v)
    void randomize_xavier() {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Xavier Normal Distribution Formula:
        // standard_deviation = sqrt(2.0 / (fan_in + fan_out))
        // In our rank 2 matrices: fan_in = rows, fan_out = cols
        float stddev = std::sqrt(2.0f / (rows + cols));
        std::normal_distribution<float> dis(0.0f, stddev);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dis(gen);
        }
    }

};

inline Tensor matmul(const Tensor& A, const Tensor& B, bool transA = false, bool transB = false) {

    size_t A_rows = transA ? A.cols : A.rows;
    size_t A_cols = transA ? A.rows : A.cols;
    size_t B_rows = transB ? B.cols : B.rows;
    size_t B_cols = transB ? B.rows : B.cols;

    if (A_cols != B_rows) {
        throw std::invalid_argument("Dimension mismatch in matmul.");
    }

    Tensor C(A_rows, B_cols);

    for (int i = 0; i < A_rows; ++i) {
        for (int k = 0; k < A_cols; ++k) {
            float a_ik = A.at(i, k);
            for (int j = 0; j < B_cols; ++j) {
                C.at(i, j) += a_ik * B.at(k, j);
            }
        }
    }

    return C;
}
