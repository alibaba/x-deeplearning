/*
 * \file concat_op.h 
 * \brief The concat operation implementation
 */
#include "blaze/operator/op/concat_op.h"

namespace blaze {

template <typename DType, int i>
struct ConcatItemHelper {
  static __inline__ __device__ int R(ConcatParam<DType>& params, size_t* offset) {
    *offset -= params.concat_item[i - 1].axis_size;
    return *offset < params.concat_item[i].axis_size ? i : ConcatItemHelper<DType, i + 1>::R(params, offset);
  }
};

template <typename DType>
struct ConcatItemHelper<DType, kMaxInputSize> {
  static __inline__ __device__ int R(ConcatParam<DType>& params, size_t* offset) {
    // NOTE: kMaxInputSize can not attained
    return 0;
  }
};

template <typename DType>
struct ConcatItemHelper<DType, 0> {
  static __inline__ __device__ int R(ConcatParam<DType>& params, size_t* offset) {
    return *offset < params.concat_item[0].axis_size ? 0 : ConcatItemHelper<DType, 1>::R(params, offset);
  }
};

template <typename DType>
__global__ void RunConcat(ConcatParam<DType> params) {
  size_t total_inner_size = params.concat_axis_size * params.inner_size;
  CUDA_KERNEL_LOOP(index, params.y_size) {
    size_t idx = index / total_inner_size;
    size_t offset = index % total_inner_size;
    int concat_item_idx = ConcatItemHelper<DType, 0>::R(params, &offset);
    DType* x = params.concat_item[concat_item_idx].x;
    offset += idx * params.concat_item[concat_item_idx].axis_size;
    params.y[index] = x[offset];
  }
}

template <>
bool ConcatOp<CUDAContext>::RunOnDevice() {
  // Check the validity of Concat
  CheckValid();

  Blob* x0 = this->Input(0);

  TYPE_SWITCH_ON_CUDA(x0->data_type(), DType, {

  ConcatParam<DType> params;
  // Prepare params and reshape y.
  Setup<DType>(&params);

  // Start to execute concat kernel
  dim3 grid, block;
  block.x = GetThreadsNum(params.y_size);
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(params.y_size, block.x));
  cudaStream_t stream = this->context_.cuda_stream();

  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&RunConcat<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });  // TYPE_SWITCH(x0->data_type(), DType, {
  return true;
}

REGISTER_CUDA_OPERATOR(Concat, ConcatOp<CUDAContext>);

}  // namespace blaze

