/*
 * \file reduce.h
 * \brief The reduce device kernel
 */
#pragma once

#include "blaze/common/blob.h"
#include "blaze/common/context.h"
#include "blaze/common/exception.h"

namespace blaze {

template <typename DType, class Context>
void ReduceSum(const DType* x,
               const int outer_size,
               const int dim,
               const int inner_size,
               DType* y,
               Context* ctx);

#ifndef INSTANTIATE_REDUCESUM
#define INSTANTIATE_REDUCESUM(T, Context)                                 \
    template <>                                                           \
    void ReduceSum<T, Context>(const T* x,                                \
                               const int outer_size,                      \
                               const int dim,                             \
                               const int inner_size,                      \
                               T* y,                                      \
                               Context* ctx) {                            \
      ReduceSumImpl<T>(x, outer_size, dim, inner_size, y, ctx);           \
    }
#endif

}  // namespace blaze

