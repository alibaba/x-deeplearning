/*
 * \file gemm_op.cc 
 * \brief The gemm operation
 */
#include "blaze/operator/op/gemm_op.h"

#include <math.h>

namespace blaze {

REGISTER_CPU_OPERATOR(Gemm, GemmOp<CPUContext>);

// Input: A, W, Bias(Optional) Output: C
OPERATOR_SCHEMA(Gemm)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .CostInferenceFunction([](const OperatorDef& def,
                              const std::vector<TensorShape>& input_shape,
                              const std::vector<DataType>& input_type,
                              const std::vector<TensorShape>& output_shape,
                              const std::vector<DataType>& output_type) {
      OpSchema::Cost cost;
      std::vector<size_t> isize;
      for (const auto& ts : input_shape) {
        size_t z = 1;
        for (const auto dim : ts.dims()) z *= dim;
        isize.push_back(z);
      }
      std::vector<size_t> osize;
      for (const auto& ts : output_shape) {
        size_t z = 1;
        for (const auto& dim : ts.dims()) z *= dim;
        osize.push_back(z);
      }

      size_t k = sqrtf((isize[0] * isize[1]) / osize[0]);
      cost.flops = 2 * k * osize[0];
      for (auto sz : isize) cost.bytes_read += sz;
      for (auto sz : osize) cost.bytes_written += sz;
      return cost;
    })
    .SetDoc(R"DOC(
Gemm operator C=A*B+Bias.
    )DOC");

}  // namespace blaze
