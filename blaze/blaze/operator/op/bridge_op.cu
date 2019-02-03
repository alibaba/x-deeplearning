/*
 * \file bridge_op.cu 
 * \brief The bridge operation for copying data from cpu to gpu 
 */

#include "blaze/operator/op/bridge_op.h"

namespace blaze {

template <>
bool BridgeOp<CUDAContext>::RunOnDevice() {
  CheckValid();
  Setup();

  Blob* x = this->Input(0);
  Blob* y = this->Output(0);
 
  // NOTE: y has been reshaped before 
  cudaStream_t stream = this->context_.cuda_stream();
  CUDA_CHECK(cudaMemcpyAsync(y->as<char>(), x->as<char>(),
                  x->size() * DataTypeSize(x->data_type()),
                  cudaMemcpyHostToDevice, stream));

  return true; 
}

REGISTER_CUDA_OPERATOR(Bridge, BridgeOp<CUDAContext>);

OPERATOR_SCHEMA(Bridge)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalShapeOfInput(0)
    .SetDoc(R"DOC(
Bridge between CPU and GPU.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

} // namespace blaze
