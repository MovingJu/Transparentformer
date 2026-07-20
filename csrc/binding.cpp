// C 트랙 전체를 위한 "얇은" glue. torch::Tensor 언패킹 + pybind11 등록만 여기서 하고,
// 실제 계산은 전부 kernels/(cabin 프로젝트, 순수 C 스타일)에 위임한다.
//
// 왜 이렇게 나눴나: <torch/extension.h>(torch::Tensor, pybind11)는 C++ 전용이라 이 경계는
// C++ 없이 못 짠다. 하지만 실제 커널 로직은 순수 C/C 스타일로 kernels/에 두고, cabin으로
// 독립적으로 빌드·테스트·포맷팅한다(사용자가 cpp보다 이쪽 워크플로우를 선호해서 채택).
#include <torch/extension.h>
#include "kernels/kernels.h"

namespace {

void check_2d_f32_contig(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    check_2d_f32_contig(a, "a");
    check_2d_f32_contig(b, "b");
    TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch for matmul");
    auto out = torch::empty({a.size(0), b.size(1)}, a.options());
    matmul_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                    a.size(0), a.size(1), b.size(1));
    return out;
}

torch::Tensor softmax(torch::Tensor x) {
    check_2d_f32_contig(x, "x");
    auto out = torch::empty_like(x);
    softmax_forward(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1));
    return out;
}

torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    check_2d_f32_contig(q, "q");
    check_2d_f32_contig(k, "k");
    check_2d_f32_contig(v, "v");
    auto out = torch::empty({q.size(0), v.size(1)}, q.options());
    sdpa_forward(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                 out.data_ptr<float>(), q.size(0), q.size(1), v.size(1),
                 static_cast<float>(scale));
    return out;
}

std::vector<torch::Tensor> layer_norm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                       double eps) {
    check_2d_f32_contig(x, "x");
    auto out = torch::empty_like(x);
    auto mean = torch::empty({x.size(0)}, x.options());
    auto rstd = torch::empty({x.size(0)}, x.options());
    layernorm_forward(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                       out.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                       x.size(0), x.size(1), static_cast<float>(eps));
    return {out, mean, rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "C kernel: matmul forward (이슈 #3)");
    m.def("softmax", &softmax, "C kernel: softmax forward (이슈 #6)");
    m.def("sdpa", &sdpa, "C kernel: SDPA forward (이슈 #8)");
    m.def("layer_norm", &layer_norm, "C kernel: LayerNorm forward (이슈 #12)");
    // sdpa_backward, layernorm_backward는 이슈 #12/#20에서 autograd.Function 만들 때
    // 필요한 시점에 여기 바인딩을 추가하면 됨 (지금은 헤더/스텁만 준비됨).
}
