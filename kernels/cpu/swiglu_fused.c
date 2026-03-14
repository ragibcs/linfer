#include <math.h>

static inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void swiglu_fused(const float* gate, const float* up, float* out, int len) {
    for (int i = 0; i < len; i++) {
        float g = gate[i];
        float swish = g * sigmoidf_fast(g);
        out[i] = swish * up[i];
    }
}
