/*
 * \file memory.h
 * \brief common memcpy util func
 */

#ifndef BLAZE_MATH_MEMORY_H_
#define BLAZE_MATH_MEMORY_H_

#include <cstddef>

namespace blaze {

struct SliceParam {
  SliceParam(size_t src_idx, size_t dst_idx, size_t step_size)
    : src_idx(src_idx), dst_idx(dst_idx), step_size(step_size)
    {}

  size_t src_idx;
  size_t dst_idx;
  size_t step_size;
};

template <typename DType, class Context>
void SliceMemcpy(DType* dst, size_t dpitch,
    const DType* src, size_t spitch,
    const SliceParam* slice_param,
    size_t count, size_t height, const Context* context); 

#ifndef INSTANTIATE_SLICEMEMCPY
#define INSTANTIATE_SLICEMEMCPY(DType, Context)               \
  template <>                                                 \
  void SliceMemcpy<DType, Context>(DType* dst, size_t dpitch, \
      const DType* src, size_t spitch,                        \
      const SliceParam* slice_param,                          \
      size_t count, size_t height, const Context* context) {  \
    SliceMemcpyImpl<DType>(dst, dpitch, src, spitch,          \
        slice_param, count, height, context);                 \
  }
#endif

template <class Context>
void Memcpy(void* dst, const void* src, size_t size,
    const Context* context);

} // namespace blaze

#endif  // BLAZE_MATH_MEMORY_H_
