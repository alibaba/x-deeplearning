/*
 * \file constant_fill_op.cc 
 * \brief The constant fill operation
 */
#include "blaze/operator/op/constant_fill_op.h"

#include "blaze/math/gemm.h"

namespace blaze {

template <>
void ConstantFillOp<CPUContext>::CopyData(void* dst, const void* src, size_t size) {
  g_copy_function[kCPU][kCPU](dst, src, 0, 0, size, 0);
}

REGISTER_CPU_OPERATOR(ConstantFill, ConstantFillOp<CPUContext>);

// Input: Zero Output: 1
OPERATOR_SCHEMA(ConstantFill)
  .NumInputs(0)
  .NumOutputs(1)
  .ShapeInferenceFunction([](const OperatorDef& op_def, const std::vector<TensorShape>&) {
    ArgumentHelper argument_helper(op_def);
    std::vector<TensorShape> ret(1);
    auto shape = argument_helper.GetRepeatedArgument<TIndex>("shape");
    for (auto dim : shape) {
      ret[0].add_dims(dim);
    }
    return ret;
  })
  .SetDoc(R"DOC(
The constant variable for network.
  )DOC");

}  // namespace blaze
