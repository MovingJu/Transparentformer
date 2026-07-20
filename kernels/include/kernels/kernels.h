#pragma once

// C 트랙 커널 선언들. 정의는 순수 C 파일(src/kernels.c)에 있음 — 이 헤더는 C에서도, C++
// (csrc/binding.cpp)에서도 그대로 include할 수 있어야 해서 `extern "C"`를 __cplusplus로 감싼다.
// (`extern "C"` 자체는 C 문법이 아니라서, 이 가드 없이 .c 파일에서 이 헤더를 include하면
// 컴파일 에러가 난다.)
//
// 각 함수의 실제 구현은 src/kernels.c에 있고, 해당 이슈에서 채워 넣으면 된다:
//   matmul_forward   -> 이슈 #3
//   softmax_forward  -> 이슈 #6
//   sdpa_forward     -> 이슈 #8
//   layernorm_forward / layernorm_backward -> 이슈 #12
//   sdpa_backward    -> 이슈 #20

#ifdef __cplusplus
extern "C" {
#endif

// out(m x p) = a(m x k) @ b(k x p), 전부 row-major, contiguous
void matmul_forward(const float* a, const float* b, float* out, long m, long k, long p);

// x, out: (rows x cols), 마지막 축(cols) 기준 softmax, max-subtract 수치안정성 포함
void softmax_forward(const float* x, float* out, long rows, long cols);

// scaled dot-product attention forward. q,k: (t x d_k), v: (t x d_v), out: (t x d_v)
void sdpa_forward(const float* q, const float* k, const float* v, float* out,
                   long t, long d_k, long d_v, float scale);

// sdpa backward: grad_out(t x d_v) -> grad_q, grad_k, grad_v
void sdpa_backward(const float* q, const float* k, const float* v,
                    const float* grad_out, float* grad_q, float* grad_k, float* grad_v,
                    long t, long d_k, long d_v, float scale);

// layernorm forward: x(rows x cols) -> out, mean/rstd는 backward에서 재사용하도록 저장
void layernorm_forward(const float* x, const float* gamma, const float* beta, float* out,
                        float* mean, float* rstd, long rows, long cols, float eps);

// layernorm backward: grad_out -> grad_x, grad_gamma, grad_beta
void layernorm_backward(const float* x, const float* gamma, const float* grad_out,
                         const float* mean, const float* rstd,
                         float* grad_x, float* grad_gamma, float* grad_beta,
                         long rows, long cols);

#ifdef __cplusplus
}  // extern "C"
#endif
