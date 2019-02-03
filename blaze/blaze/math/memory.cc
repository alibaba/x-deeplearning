/*
 * \file memory.cc
 * \brief memeory utils for cpu 
 */

#include "blaze/math/memory.h"
#include "blaze/common/context.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
void SliceMemcpyImpl(DType* dst, size_t dpitch,
    const DType* src, size_t spitch,
    const SliceParam* slice_param,
    size_t count, size_t height,
    const CPUContext* context) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < count; ++j) {
      memcpy(dst + dpitch * i + slice_param[j].dst_idx,
          src + spitch * i + slice_param[j].src_idx,
          slice_param[j].step_size * sizeof(DType));
    } 
  }
}

INSTANTIATE_SLICEMEMCPY(float16, CPUContext);
INSTANTIATE_SLICEMEMCPY(float, CPUContext)
INSTANTIATE_SLICEMEMCPY(double, CPUContext)

template <>
void Memcpy<CPUContext>(void* dst, const void* src, size_t size,
    const CPUContext* context) {
  memcpy(dst, src, size); 
} 

} // namespace blaze
