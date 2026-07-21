#include <string.h>
#include "../include/kernels/kernels.h"

static inline float *matrix_index_mut(float *mat, const long mat_col, long row, long col)
{
    return &mat[row * mat_col + col];
}
static inline const float *matrix_index(const float *mat, const long mat_col, long row, long col)
{
    return &mat[row * mat_col + col];
}
void matmul_forward(const float *a, const float *b, float *out, long m, long k, long p)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float cell = 0;
            for (int l = 0; l < k; ++l) {
                cell += *matrix_index(a, k, i, l) * *matrix_index(b, p, l, j);
            }
            *matrix_index_mut(out, p, i, j) = cell;
        }
    }
}

void softmax_forward(const float *x, float *out, long rows, long cols)
{
    // TODO(#6): max-subtract 수치안정성 포함해서 실제 softmax 구현
    (void)x;
    memset(out, 0, sizeof(float) * rows * cols);
}

void sdpa_forward(const float *q, const float *k, const float *v, float *out, long t, long d_k,
                  long d_v, float scale)
{
    // TODO(#8): matmul_forward + softmax_forward를 이용해 SDPA forward 구현
    (void)q;
    (void)k;
    (void)v;
    (void)d_k;
    (void)scale;
    memset(out, 0, sizeof(float) * t * d_v);
}

void sdpa_backward(const float *q, const float *k, const float *v, const float *grad_out,
                   float *grad_q, float *grad_k, float *grad_v, long t, long d_k, long d_v,
                   float scale)
{
    // TODO(#20): SDPA backward 손미분 구현 (C 트랙 최종 보스)
    (void)q;
    (void)k;
    (void)v;
    (void)grad_out;
    (void)d_v;
    (void)scale;
    memset(grad_q, 0, sizeof(float) * t * d_k);
    memset(grad_k, 0, sizeof(float) * t * d_k);
    memset(grad_v, 0, sizeof(float) * t * d_k);
}

void layernorm_forward(const float *x, const float *gamma, const float *beta, float *out,
                       float *mean, float *rstd, long rows, long cols, float eps)
{
    // TODO(#12): LayerNorm forward 구현 (mean/rstd를 backward용으로 저장)
    (void)x;
    (void)gamma;
    (void)beta;
    (void)eps;
    memset(out, 0, sizeof(float) * rows * cols);
    memset(mean, 0, sizeof(float) * rows);
    memset(rstd, 0, sizeof(float) * rows);
}

void layernorm_backward(const float *x, const float *gamma, const float *grad_out,
                        const float *mean, const float *rstd, float *grad_x, float *grad_gamma,
                        float *grad_beta, long rows, long cols)
{
    // TODO(#12): LayerNorm backward 손미분 구현
    (void)x;
    (void)gamma;
    (void)grad_out;
    (void)mean;
    (void)rstd;
    memset(grad_x, 0, sizeof(float) * rows * cols);
    memset(grad_gamma, 0, sizeof(float) * cols);
    memset(grad_beta, 0, sizeof(float) * cols);
}
