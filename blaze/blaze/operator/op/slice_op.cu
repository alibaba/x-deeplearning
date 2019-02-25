/*
 * \file slice_op.cu 
 * \brief The slice operation
 */
#include "blaze/operator/op/slice_op.h"

namespace blaze {

template <typename DType>
__global__ void RunSlice(SliceParam<DType> params) {
  size_t width = params.sci.end - params.sci.start;
  size_t count = width * params.sci.inner_size * params.sci.outer_size;
  size_t const_off = params.sci.start * params.sci.inner_size; 
  size_t const_width = (params.sci.size - width) * params.sci.inner_size;
  
  CUDA_KERNEL_LOOP(index, count) {
    size_t out_idx = index / (width * params.sci.inner_size);
    size_t src_index = index + out_idx * const_width + const_off;
    params.y[index] = params.x[src_index];
  }
}

template <>
bool SliceOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);

  // Check the valid of SliceOp
  CheckValid();

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {

  SliceParam<DType> params;
  Setup<DType>(&params);
  
  // launch the kernel
  dim3 grid, block;
  size_t count = params.sci.outer_size * (params.sci.end - params.sci.start) * params.sci.inner_size;
  block.x = GetThreadsNum(count);
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(count, block.x));
  cudaStream_t stream = this->context_.cuda_stream();
 
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&RunSlice<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });  // TYPE_SWITCH(X->data_type(), DType, {
  return true;
}

REGISTER_CUDA_OPERATOR(Slice, SliceOp<CUDAContext>);

}  // namespace blaze
