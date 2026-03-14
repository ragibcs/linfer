#include <immintrin.h>
#include <omp.h>

void gemm_f32_avx2(
    const float* restrict a,
    const float* restrict b,
    float* restrict c,
    int m,
    int n,
    int k
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int x = 0; x < k; x++) {
                acc += a[i * k + x] * b[x * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}
