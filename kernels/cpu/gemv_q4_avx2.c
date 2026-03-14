#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

#include "utils/simd_helpers.h"

void gemv_q4_avx2(
    const uint8_t* restrict weights,
    const float* restrict scales,
    const float* restrict input,
    float* restrict output,
    int rows,
    int cols
) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float acc_scalar = 0.0f;
        const uint8_t* row_w = weights + (r * cols) / 2;
        const float* row_sc = scales + (r * cols) / 32;

        for (int c = 0; c < cols; c += 32) {
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < 16; i++) {
                uint8_t packed = row_w[(c / 2) + i];
                int8_t q0 = (int8_t)((packed & 0x0F) - 8);
                int8_t q1 = (int8_t)(((packed >> 4) & 0x0F) - 8);
                float dq0 = (float)q0 * row_sc[c / 32];
                float dq1 = (float)q1 * row_sc[c / 32];

                float tmp[8] = {dq0, dq1, 0, 0, 0, 0, 0, 0};
                float in[8] = {
                    input[c + i * 2 + 0], input[c + i * 2 + 1],
                    0, 0, 0, 0, 0, 0
                };
                __m256 wv = _mm256_loadu_ps(tmp);
                __m256 iv = _mm256_loadu_ps(in);
                acc = _mm256_fmadd_ps(wv, iv, acc);
            }
            acc_scalar += horizontal_sum(acc);
        }

        output[r] = acc_scalar;
    }
}
