/*
 * \file split_op.cu
 * \brief The split operation on cpu
 */
#include "blaze/operator/op/split_op.h"

namespace blaze {

template <typename DType>
__inline__ __device__ bool IsEqualSplit(const SplitParam<DType>& params) {
  bool is_equal = true;
  for (size_t k = 1; k < params.split_num; ++k) {
    if (params.split_item[k].split_axis_size != params.split_item[0].split_axis_size) {
      is_equal = false;
      break;
    }
  }
  return is_equal;
}

template <typename DType>
__inline__ __device__ void RunEqualSplit(SplitParam<DType>& params) {
  size_t z = params.inner_size * params.axis_size;
  size_t size = params.outer_size * z;
  size_t split_axis_size = params.split_item[0].split_axis_size;
  CUDA_KERNEL_LOOP(index, size) {
    size_t outer_index = index / z;
    size_t j = index % z;
    size_t split_index = j / split_axis_size;
    size_t split_offset = j % split_axis_size;
    params.split_item[split_index].y[outer_index * split_axis_size + split_offset] = params.x[index];
  }
}

template <typename DType>
__global__ void RunSplit(SplitParam<DType> params) {
  if (IsEqualSplit(params)) {
    RunEqualSplit(params);
  } else {
    //TODO:
  }
}

template <>
bool SplitOp<CUDAContext>::RunOnDevice() {
  if (auto_split_ && axis_ == 0) {
    return RunOnDevice_SplitAxis0();
  }
  // Check the validity of Split
  CheckValid();
  Blob* x = this->Input(0);

  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {

  SplitParam<DType> params;
  // Prepare params and reshape y
  Setup<DType>(&params);

  // Start to execute split kernel
  dim3 grid, block;
  block.x = GetThreadsNum(x->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(x->size(), block.x));
  cudaStream_t stream = this->context_.cuda_stream();

  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&RunSplit<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });
  return true;
}

REGISTER_CUDA_OPERATOR(Split, SplitOp<CUDAContext>);

}  // namespace blaze
