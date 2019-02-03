/*
 * \file gather_op.cu
 * \brief The gather operation on cpu
 */
#include "blaze/operator/op/gather_op.h"

namespace blaze {

template <typename ValueType, typename IDType>
__global__ void GatherKernel(GatherParam<ValueType, IDType> params) {
  CUDA_KERNEL_LOOP(index, params.y_size) {
    size_t outer_index = index / params.y_inner_size;
    size_t inner_offset = index % params.y_inner_size;
    size_t axis_index = params.indices[inner_offset / params.inner_size];
    size_t offset = inner_offset % params.inner_size;

    size_t src_offset = outer_index * params.axis_size * params.inner_size +
        axis_index * params.inner_size + offset;
    params.y[index] = params.data[src_offset];
  }
}

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  // Check the validity of Split
  CheckValid();

  Blob* data = this->Input(0);
  Blob* indices = this->Input(1);

  TYPE_SWITCH_ON_CUDA(data->data_type(), ValueType, {
  ID_TYPE_SWITCH(indices->data_type(), IDType, {

  GatherParam<ValueType, IDType> params;
  // Prepare params and reshape
  Setup<ValueType, IDType>(&params);

  // Start to execute gather kernel
  dim3 grid, block;
  block.x = GetThreadsNum(params.y_size);
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(params.y_size, block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&GatherKernel<ValueType, IDType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));
  });
  });
  return true;
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);

}  // namespace blaze

