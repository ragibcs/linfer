#include <math.h>

void sdpa_f32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int seq,
    int dim
) {
    for (int i = 0; i < dim; i++) {
        out[i] = 0.0f;
    }

    float scale = 1.0f / sqrtf((float)dim);
    for (int t = 0; t < seq; t++) {
        float score = 0.0f;
        for (int d = 0; d < dim; d++) {
            score += q[d] * k[t * dim + d];
        }
        float w = expf(score * scale);
        for (int d = 0; d < dim; d++) {
            out[d] += w * v[t * dim + d];
        }
    }
}
