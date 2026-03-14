#include <math.h>

void rope_f32(float* q, float* k, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float ang = (float)pos * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);

        float q0 = q[i], q1 = q[i + 1];
        float k0 = k[i], k1 = k[i + 1];

        q[i] = q0 * cs - q1 * sn;
        q[i + 1] = q0 * sn + q1 * cs;
        k[i] = k0 * cs - k1 * sn;
        k[i + 1] = k0 * sn + k1 * cs;
    }
}
