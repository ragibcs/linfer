#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemv_q8_avx2(
    const int8_t* restrict weights,
    const float* restrict scales,
    const float* restrict input,
    float* restrict output,
    int rows,
    int cols
) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const int8_t* row = weights + (r * cols);
        float scale = scales[r];

        for (int c = 0; c < cols; c++) {
            acc += (float)row[c] * scale * input[c];
        }
        output[r] = acc;
    }
}
