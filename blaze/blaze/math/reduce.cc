/*
 * \file reduce.cc
 * \brief The reduce device kernel
 */
#include "blaze/math/reduce.h"

#include "blaze/math/vml.h"
#include <immintrin.h>

namespace blaze {

template <typename DType>
static void ReduceSumImpl(const DType* x,
                          const int outer_size,
                          const int dim,
                          const int inner_size,
                          DType* y,
                          CPUContext* ctx) {
  for (int i = 0; i < outer_size; ++i) {
    memset(y, 0, sizeof(DType) * inner_size);
    for (int j = 0; j < dim; j ++) {
      for (int k = 0; k < inner_size; ++k) {
        y[k] += x[k];
      }
      x += inner_size;
    }
    y += inner_size;
  }
}

INSTANTIATE_REDUCESUM(float, CPUContext)
INSTANTIATE_REDUCESUM(double, CPUContext)

}  // namespace blaze

