#include <iostream>
#include "tensor.hpp"
#include <algorithm>
#include <vector>

int main() {
    // Hyperparameters
    size_t S = 256;
    size_t N = 50;
    size_t d = 64;

    // Data tensors
    Tensor X_thermal(S, d); // (S, d)
    Tensor X_rgb(S, d); // (S, d)
    Tensor X_mmwave(N, d); // (N, d)
    Tensor Target(S, d); // (S, d)

    // Learnable weight matrices
    Tensor W_fuse(2 * d, d); // (2d, d)
    Tensor W_q(d, d); // (d, d)
    Tensor W_k(d, d); // (d, d)
    Tensor W_v(d, d); // (d, d)

    // Initialize Physics/Sensor Tensors
    X_thermal.randomize_uniform();
    X_rgb.randomize_uniform();
    X_mmwave.randomize_uniform();
    Target.randomize_uniform();

    // Initialize Mathematical Transformations
    W_fuse.randomize_xavier();
    W_q.randomize_xavier();
    W_k.randomize_xavier();
    W_v.randomize_xavier();

    float learning_rate = 0.001f;

    Tensor X_spatial(S, 2 * d); // (S, 2d)

    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // Training loop

    for (int epoch = 0; epoch < 1000; ++epoch) {
        // --- FORWARD PASS ---

        // Concatenate X_thermal and X_rgb

        for (size_t i = 0; i < S; ++i) {
            float *dest_thermal = &X_spatial.data[i * (2 * d)];
            float *dest_rgb = &X_spatial.data[i * (2 * d) + d];

            const float *src_thermal = &X_thermal.data[i * d];
            const float *src_rgb = &X_thermal.data[i * d];

            std::copy(src_thermal, src_thermal + d, dest_thermal);
            std::copy(src_rgb, src_rgb + d, dest_rgb);
        }

        Tensor Q_base = matmul(X_spatial, W_fuse); // (S, 2d) * (2d, d) = (S, d)
        Tensor Q = matmul(Q_base, W_q); // (S, d) * (d, d) = (S, d)
        Tensor K = matmul(X_mmwave, W_k); // (N, d) * (d, d) = (N, d)
        Tensor V = matmul(X_mmwave, W_v); // (N, d) * (d, d) = (N, d)

        // Attention = Softmax(Q * K^T / sqrt(d)) * V
        Tensor Scores = matmul(Q, K, false, true); // (S, d) * (d, N) = (S, N)

        for (size_t i = 0; i < Scores.data.size(); ++i) {
            Scores.data[i] *= scale;
        }

        // Softmax(Q * K^T / sqrt(d))
        for (size_t i = 0; i < S; ++i) {
            float *row = &Scores.data[i * N]; // getting i-th row

            // finding max(x)
            float max = row[0];
            for (size_t j = 1; j < N; ++j) {
                if (row[j] > max) {
                    max = row[j];
                }
            }

            // find summation e^(xj - max)
            float exp_sum = 0.0f;
            for (size_t j = 0; j < N; ++j) {
                row[j] = std::exp(row[j] - max);
                exp_sum += row[j];
            }

            // performing e^(xi - max) / summmation e^(xj - max)
            float reciprocal = 1.0f / exp_sum;
            for (size_t j = 0; j < N; ++j) {
                row[j] *= reciprocal;
            }

        }

        // softmax * V
        Tensor Final_Out = matmul(Scores, V);

        // --- LOSS CALCULATION ---
        int M = S * d;
        float loss = 0.0f;

        for (size_t i = 0; i < M; ++i) {
            loss += (Final_Out.data[i] - Target.data[i]) * 2;
        }

        loss /= M;

        std::cout << loss << '\n';

        // --- BACKWARD PASS ---
        // ... (Execute the gradient splits) ...

        // --- OPTIMIZER STEP ---
        // ... (Update W_fuse, W_q, W_k, W_v using learning_rate) ...
    }

    return 0;

}
