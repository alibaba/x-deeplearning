/*
 * \file gru_op.cc
 * \brief The gru operation
 */
#include "blaze/math/gemm.h"
#include "blaze/math/vml.h"
#include "blaze/operator/op/gru_op.h"

namespace blaze {

template <typename DType, typename Context>
inline void VML_Sigmoid(const size_t n, const DType* x, DType* y,
                        Context* ctx) {
  for (int i = 0; i < n; i++) { y[i] = -x[i]; }
  VML_Exp<DType, Context>(n, y, y, ctx);
  for (int i = 0; i < n; i++) { y[i] = 1.0 / (1 + y[i]); }
}

template <typename DType, typename Context>
inline void VML_AddMul(const size_t n, const DType* a, const DType* b,
                       const DType* c, DType* z, Context* ctx) {
  for (int i = 0; i < n; i++) { z[i] = a[i] + b[i] * c[i]; }
}

template <typename DType>
inline void SetOutput(const size_t n, const DType* z, const DType* h,
                      const DType* last_output, DType* output) {
  for (int i = 0; i < n; i++) {
    output[i] = h[i] + z[i] * (last_output[i] - h[i]);
  }
}

template <typename DType, typename Context>
void GRUKernel(const GRUParam<DType>& params, Context* ctx) {
  const auto batch_size = params.batch_size;
  const auto round = params.round;
  const auto elts = params.elts;
  if (round <= 0) { return; }
  Gemm<DType, Context>(CblasNoTrans, CblasNoTrans,
                       batch_size * round, elts * 3, elts, 1.0,
                       params.x, params.i2h, 0.0, params.preact, ctx);
  memset(params.y, 0, batch_size * round * elts * sizeof(DType));
  auto act = params.act;
  const auto h2h = params.h2h;
  const auto h2h_bias = params.h2h_bias;
  const auto i2h_bias = params.i2h_bias;
  DType alpha = 1.0;
  DType beta = 0.0;
  for (int b = 0; b < batch_size; b++) {
    DType* y = &params.y[b * round * elts];
    DType* preact = &params.preact[b * round * elts * 3];
    bool preact_nonzero = false;
    for (int i = 0; i < round; i++) {
      // Recheck zeroness in this round if the check failed in the
      // last round
      if (!preact_nonzero) {
        for (int k = 0; k < elts; k++) {
          if (preact[k] != 0) {
            preact_nonzero = true;
            break;
          }
        }
      }
      if (preact_nonzero) {
        if (i > 0) {
          DType* prev_y = y - elts;
          Gemm(CblasNoTrans, CblasNoTrans, 1, elts * 3, elts, 1.0,
               prev_y, h2h, 0.0, act, ctx);
          VML_Add(elts * 3, act, h2h_bias, act, ctx);
          VML_Add(elts * 3, preact, i2h_bias, preact, ctx);
          VML_Add(elts * 2, preact, act, preact, ctx);
          VML_Sigmoid(elts * 2, preact, preact, ctx);
          VML_AddMul(elts, &preact[elts * 2], preact, &act[elts * 2],
                     &preact[elts * 2], ctx);
          VML_Tanh(elts, &preact[elts * 2], &preact[elts * 2], ctx);
          SetOutput(elts, &preact[elts], &preact[elts * 2], prev_y, y);
        } else {
          VML_Add(elts * 3, preact, i2h_bias, preact, ctx);
          VML_Add(elts * 2, preact, h2h_bias, preact, ctx);
          VML_Sigmoid(elts * 2, preact, preact, ctx);
          VML_AddMul(elts, &preact[elts * 2], preact, &h2h_bias[elts * 2],
                     &preact[elts * 2], ctx);
          VML_Tanh(elts, &preact[elts * 2], &preact[elts * 2], ctx);
          SetOutput(elts, &preact[elts], &preact[elts * 2], y, y);
        }
      } else {
        memset(y, 0, elts * sizeof(DType));
      }
      preact += elts * 3;
      y += elts;
    }
  }
}

template <>
bool GRUOp<CPUContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  TYPE_SWITCH(x->data_type(), DType, {
    GRUParam<DType> params;
    Setup(&params);
    GRUKernel<DType, CPUContext>(params, &context_);
  });
  return true;
}

REGISTER_CPU_OPERATOR(GRU, GRUOp<CPUContext>);

// For ONNX: Input: X, W, R, B Output: Y
// For Ulf: Input: X, h2hweight, i2hweight, h2hBias, i2hbias
OPERATOR_SCHEMA(GRU)
    .NumInputs(3, 5)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Cast the input to another type Tensor.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze
