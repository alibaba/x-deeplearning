/*
 * \file fused_slice_concat_op.cu 
 * \brief The fused slice and concat operation
 */
#include "blaze/operator/fused_op/fused_slice_concat_op.h"

namespace blaze {

__device__ __inline__ void GetSliceItemInfo(
        const SliceItem* slice_item,
        size_t concat_input_size,
        size_t axis_idx,
        size_t* slice_item_index,
        size_t* off) {
  size_t i = 0;
  for (; i < concat_input_size; ++i) {
    if (axis_idx >= slice_item[i].len) axis_idx -= slice_item[i].len;
    else break;
  }
  *slice_item_index = i;
  *off = axis_idx;
}

// SliceAxis == ConcatAxis
template <typename DType>
__device__ __inline__ void FusedSliceConcatEqual(const FusedSliceConcatParam<DType>& params) {
  size_t size0 = params.concat_axis_size * params.concat_inner_size;
  size_t outer_idx, axis_idx, inner_idx;
  // [Outer, Axis(Concat/Slice), Inner]
  CUDA_KERNEL_LOOP(index, params.y_size) {
    // The concat outer index
    outer_idx = index / size0;
    auto remaining = index % size0;
    // The axis_idx is the axis's position, and inner_idx is the inner's position.
    axis_idx = remaining / params.concat_inner_size;
    inner_idx = remaining % params.concat_inner_size;
    size_t slice_item_index, off;
    // TODO: Can optimize the function, when all the slice item's len are equal.
    GetSliceItemInfo(params.slice_item, params.concat_input_size, axis_idx, &slice_item_index, &off);
    // Calculate the x' offset
    auto x_offset = outer_idx * size0 +
        (params.slice_item[slice_item_index].start + off) * params.concat_inner_size +
        inner_idx;
    params.y[index] = params.x[x_offset];
  }
}

// SliceAxis < ConcatAxis
template <typename DType>
__device__ __inline__ void FusedSliceConcatLesser(const FusedSliceConcatParam<DType>& params) {
  size_t raw_concat_axis_size = params.concat_axis_size / params.concat_input_size;
  
  size_t slice_outer_size = params.slice_outer_size;
  size_t slice_axis_size = params.slice_item[0].len;
  size_t middle_size = params.slice_inner_size / (params.concat_inner_size * raw_concat_axis_size);
  size_t concat_axis_size = params.concat_axis_size;
  size_t concat_inner_size = params.concat_inner_size;

  size_t slice_outer_index, slice_axis_index, middle_index, concat_axis_index, concat_inner_index;
  size_t slice_item_index, pos;

  size_t len1, len2, len3, len4;
  len1 = concat_inner_size;
  len2 = len1 * raw_concat_axis_size;
  len3 = len2 * middle_size;
  len4 = len3 * params.slice_axis_size;

  // [SliceOuter, SliceAxis, Middle, ConcatAxis, ConcatInner]
  CUDA_KERNEL_LOOP(index, params.y_size) {
    pos = index;
    concat_inner_index = pos % concat_inner_size;  pos /= concat_inner_size;
    concat_axis_index  = pos % concat_axis_size;   pos /= concat_axis_size;
    middle_index       = pos % middle_size;        pos /= middle_size;
    slice_axis_index   = pos % slice_axis_size;    pos /= slice_axis_size;
    slice_outer_index  = pos % slice_outer_size;

    slice_item_index   = concat_axis_index / raw_concat_axis_size;
    concat_axis_index  = concat_axis_index % raw_concat_axis_size;
    slice_axis_index   = params.slice_item[slice_item_index].start + slice_axis_index;

    pos = slice_outer_index * len4 +
          slice_axis_index * len3 +
          middle_index * len2 +
          concat_axis_index * len1 +
          concat_inner_index;
    params.y[index] = params.x[pos];
  }
}

template <typename DType>
__device__ __inline__ void FusedSliceConcatGreater(const FusedSliceConcatParam<DType>& params) {
  size_t raw_concat_axis_size = params.concat_axis_size / params.concat_input_size;

  size_t concat_outer_size = params.concat_outer_size;
  size_t concat_axis_size  = params.concat_axis_size;
  size_t middle_size       = params.concat_inner_size / (params.slice_item[0].len * params.slice_inner_size);
  size_t slice_axis_size   = params.slice_item[0].len;
  size_t slice_inner_size  = params.slice_inner_size;

  size_t concat_outer_index, concat_axis_index, middle_index, slice_axis_index, slice_inner_index;
  size_t slice_item_index, pos;

  size_t len1, len2, len3, len4;
  len1 = slice_inner_size;
  len2 = len1 * params.slice_axis_size;
  len3 = len2 * middle_size;
  len4 = len3 * raw_concat_axis_size;

  // [ConcatOuter, ConcatAxis, Middle, SliceAxis, SliceInner]
  CUDA_KERNEL_LOOP(index, params.y_size) {
    pos = index;
    slice_inner_index  = pos % slice_inner_size;   pos /= slice_inner_size;
    slice_axis_index   = pos % slice_axis_size;    pos /= slice_axis_size;
    middle_index       = pos % middle_size;        pos /= middle_size;
    concat_axis_index  = pos % concat_axis_size;   pos /= concat_axis_size;
    concat_outer_index = pos % concat_outer_size;

    slice_item_index   = concat_axis_index / raw_concat_axis_size;
    concat_axis_index  = concat_axis_index % raw_concat_axis_size;
    slice_axis_index   = params.slice_item[slice_item_index].start + slice_axis_index;

    pos = concat_outer_index * len4 +
          concat_axis_index * len3 +
          middle_index * len2 +
          slice_axis_index * len1 +
          slice_inner_index;
    params.y[index] = params.x[pos];
  }
}

template <typename DType>
__global__ void RunFusedSliceConcat(FusedSliceConcatParam<DType> params) {
  if (params.slice_axis == params.concat_axis) {
    FusedSliceConcatEqual(params);
  } else if (params.slice_axis < params.concat_axis) {
    FusedSliceConcatLesser(params);
  } else {
    FusedSliceConcatGreater(params);
  }
}

template <>
bool FusedSliceConcatOp<CUDAContext>::RunOnDevice() {
  // check th validity of FusedSliceConcat
  CheckValid();

  Blob* x0 = this->Input(0);
  TYPE_SWITCH_ON_CUDA(x0->data_type(), DType, {
  
  FusedSliceConcatParam<DType> params;
  // Prepare params and reshape
  Setup<DType>(&params);
  
  // Start to execute slice/concat fused kernel
  dim3 grid, block;
  block.x = GetThreadsNum(params.y_size);
  grid.x = CUDA_GET_BLOCKS(params.y_size, block.x);
  cudaStream_t stream = this->context_.cuda_stream();

  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&RunFusedSliceConcat<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });
  return true;
}

REGISTER_CUDA_OPERATOR(FusedSliceConcat, FusedSliceConcatOp<CUDAContext>);

}  // namespace blaze
