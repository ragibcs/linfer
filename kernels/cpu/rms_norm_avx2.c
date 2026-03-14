#include <math.h>

void rms_norm_avx2(const float* input, const float* weight, float* output, int len, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < len; i++) {
        ss += input[i] * input[i];
    }
    float inv = 1.0f / sqrtf(ss / (float)len + eps);
    for (int i = 0; i < len; i++) {
        output[i] = input[i] * inv * weight[i];
    }
}
