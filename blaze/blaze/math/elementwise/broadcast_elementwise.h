/*
 * \file broadcast_elementwise.h
 * \desc the broadcast implementation of ElementwiseOp
 */
#pragma once

#include "blaze/common/exception.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/log.h"

namespace blaze {
namespace broadcast {

const int MAX_DIM = 5;
#define DEEPNET_BROADCAST_NDIM_SWITCH(ndim, NDim, ...)  \
  if (ndim <= 2) {                    \
    const int NDim = 2;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= 4) {             \
    const int NDim = 4;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= MAX_DIM) {       \
    const int NDim = MAX_DIM;         \
    {__VA_ARGS__}                     \
  } else {                            \
    BLAZE_CONDITION_THROW("NDim ", ndim,  \
      " is too large.");              \
  }

template<int ndim>
struct Shape {
  TIndex shape[ndim];
  BLAZE_INLINE_X Shape() {}
  Shape(const std::vector<TIndex>& vec) {
    if (vec.size() > ndim) {
      printf("[FATAL]vec size %d larger than ndim %d\n", vec.size(), ndim);
      *(int*) nullptr = 1;
    }
    memset(shape, 0, ndim * sizeof(TIndex));
    memcpy(shape, vec.data(), vec.size() * sizeof(TIndex));
  }
  BLAZE_INLINE_X TIndex& operator[](int idx) {
    return shape[idx];
  }
  BLAZE_INLINE_X const TIndex& operator[](int idx) const {
    return shape[idx];
  }
#ifndef NDEBUG
  BLAZE_INLINE_X void Debug(const char* prefix) const {
    switch(ndim) {
      case 1:
        printf("%sshape:%lu\n", prefix, shape[0]);
        break;
      case 2:
        printf("%sshape:%lu %lu\n", prefix, shape[0], shape[1]);
        break;
      case 3:
        printf("%sshape:%lu %lu %lu\n", prefix, shape[0], shape[1], shape[2]);
        break;
      case 4:
        printf("%sshape:%lu %lu %lu %lu\n", prefix, shape[0], shape[1], shape[2], shape[3]);
        break;
      case 5:
        printf("%sshape:%lu %lu %lu %lu %lu\n", prefix, shape[0], shape[1], shape[2], shape[3], shape[4]);
        break;
      default:
        printf("%sshape:", prefix);
        for (int i = 0; i < ndim; i++) {
          printf("%lu ", shape[i]);
        }
        printf("\n");
        break;
    }
  }
#endif
};

inline int BroadcastShapeCompact(const std::vector<TIndex>& lshape,
                                 const std::vector<TIndex>& rshape,
                                 const std::vector<TIndex>& oshape,
                                 std::vector<TIndex>* new_lshape,
                                 std::vector<TIndex>* new_rshape,
                                 std::vector<TIndex>* new_oshape) {
  //if (lshape == rshape) return 0;  // comment this line, compute
  //already work on shape equal condition.
  int odim = std::max<int>(oshape.size(), broadcast::MAX_DIM);
  new_lshape->resize(odim, 1);
  new_rshape->resize(odim, 1);
  new_oshape->resize(odim, 1);
  int bl = oshape.size() - lshape.size();
  int br = oshape.size() - rshape.size();
  int j = 0;
  TIndex lprod = 1;
  TIndex rprod = 1;
  TIndex oprod = 1;
  for (int i = 0; i < oshape.size(); ++i) {
    TIndex l = 1;
    TIndex r = 1;
    TIndex o = oshape[i];
    if (i >= bl) l = lshape[i - bl];
    if (i >= br) r = rshape[i - br];
    if ((lprod != rprod || l != r) &&
        lprod * l > 1 && rprod * r > 1) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1;
      ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  
  BLAZE_CONDITION_THROW(j <= broadcast::MAX_DIM,
      "Too many broadcast dims, lshape size:", lshape.size(),
      " rshape size:", rshape.size());
  DEEPNET_BROADCAST_NDIM_SWITCH(j, NDim, {
    new_lshape->resize(NDim);
    new_rshape->resize(NDim);
    new_oshape->resize(NDim);
  });
  return j;
}

inline int BroadcastShapeCompact(const std::vector<TIndex>& cshape,
                                 const std::vector<TIndex>& lshape,
                                 const std::vector<TIndex>& rshape,
                                 const std::vector<TIndex>& oshape,
                                 std::vector<TIndex>* new_cshape,
                                 std::vector<TIndex>* new_lshape,
                                 std::vector<TIndex>* new_rshape,
                                 std::vector<TIndex>* new_oshape) {
  int odim = std::max<int>(oshape.size(), broadcast::MAX_DIM);
  new_cshape->resize(odim, 1);
  new_lshape->resize(odim, 1);
  new_rshape->resize(odim, 1);
  new_oshape->resize(odim, 1);
  int bc = oshape.size() - cshape.size();
  int bl = oshape.size() - lshape.size();
  int br = oshape.size() - rshape.size();
  int j = 0;
  TIndex cprod = 1;
  TIndex lprod = 1;
  TIndex rprod = 1;
  TIndex oprod = 1;
  for (int i = 0; i < oshape.size(); ++i) {
    TIndex c = 1;
    TIndex l = 1;
    TIndex r = 1;
    TIndex o = oshape[i];
    if (i >= bc) c = cshape[i - bc];
    if (i >= bl) l = lshape[i - bl];
    if (i >= br) r = rshape[i - br];
    if (lprod != rprod || l != r
        || lprod != cprod || l != c
        || rprod != cprod || r != c) {
      if (cprod * c > 1 && lprod * l > 1 && rprod * r > 1) {
        (*new_cshape)[j] = cprod;
        (*new_lshape)[j] = lprod;
        (*new_rshape)[j] = rprod;
        (*new_oshape)[j] = oprod;
        cprod = lprod = rprod = oprod = 1;
        ++j;
      }
    }
    cprod *= c;
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (cprod > 1 || lprod > 1 || rprod > 1) {
    (*new_cshape)[j] = cprod;
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }

  BLAZE_CONDITION_THROW(j <= broadcast::MAX_DIM,
                        "Too many broadcast dims, lshape size:", lshape.size(),
                        " rshape size:", rshape.size(),
                        " cshape size:", cshape.size());
  DEEPNET_BROADCAST_NDIM_SWITCH(j, NDim, {
    new_cshape->resize(NDim);
    new_lshape->resize(NDim);
    new_rshape->resize(NDim);
    new_oshape->resize(NDim);
  });
  return j;
}

/* Calculate stride of each dim from shape */
template<int ndim>
BLAZE_INLINE_X Shape<ndim> calc_stride(const std::vector<TIndex>& shape) {
  Shape<ndim> stride;
  TIndex cumprod = 1;
#pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}

template<int ndim>
BLAZE_INLINE_X TIndex dot(const Shape<ndim>& coord,
                       const Shape<ndim>& stride) {
  TIndex ret = 0;
#pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret += coord[i] * stride[i];
  }
  return ret;
}

template<int ndim>
BLAZE_INLINE_X void unravel(const int idx,
                            const Shape<ndim>& shape,
                            Shape<ndim>* ret) {
  int j = idx;
#pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    (*ret)[i] = j % shape[i];
    j /= shape[i];
  }
}

template<int ndim, typename DType, typename OP>
struct broadcast_kernel {
  BLAZE_INLINE_X static void Map(int idx,
                                 const Shape<ndim>& lstride,
                                 const Shape<ndim>& rstride,
                                 const Shape<ndim>& oshape,
                                 const DType* lhs,
                                 const DType* rhs,
                                 DType* out) {
    Shape<ndim> coord;
    unravel<ndim>(idx, oshape, &coord);
    auto lidx = dot<ndim>(coord, lstride);
    auto ridx = dot<ndim>(coord, rstride);

#ifndef NDEBUG
    LOG_DEBUG("idx:%d, lidx:%d, ridx:%d", idx, lidx, ridx);
#endif
    
    out[idx] = OP::Map(lhs[lidx], rhs[ridx]);

#ifndef NDEBUG
    LOG_DEBUG("[%d]%f = %f OP %f", idx, out[idx], lhs[lidx], rhs[ridx]);
#endif
  }
};

template<int ndim, typename IType, typename DType, typename OP>
struct broadcast_condition_kernel {
  BLAZE_INLINE_X static void Map(int idx,
                                 const Shape<ndim>& cstride,
                                 const Shape<ndim>& lstride,
                                 const Shape<ndim>& rstride,
                                 const Shape<ndim>& oshape,
                                 const IType* conditions,
                                 const DType* lhs,
                                 const DType* rhs,
                                 DType* out) {
    Shape<ndim> coord;
    unravel<ndim>(idx, oshape, &coord);
    auto cidx = dot<ndim>(coord, cstride);
    auto lidx = dot<ndim>(coord, lstride);
    auto ridx = dot<ndim>(coord, rstride);

#ifndef NDEBUG
    LOG_DEBUG("idx:%d, cidx:%d, lidx:%d, ridx:%d", idx, cidx, lidx, ridx);
#endif

    out[idx] = OP::Map(conditions[cidx], lhs[lidx], rhs[ridx]);

#ifndef NDEBUG
    LOG_DEBUG("[%d]%f = %d %f OP %f", idx, out[idx], conditions[cidx], lhs[lidx], rhs[ridx]);
#endif
  }
};

template<typename DType, typename OP, template<typename, typename> class KernelLauncher,
    class Context>
bool BroadcastCompute(const DType* input1,
                      const std::vector<TIndex>& shape1,
                      const DType* input2,
                      const std::vector<TIndex>& shape2,
                      DType* output,
                      const std::vector<TIndex>& outshape,
                      const Context& context) {
  std::vector<TIndex> new_lshape, new_rshape, new_oshape;
  int ndim = BroadcastShapeCompact(shape1,
                                   shape2,
                                   outshape,
                                   &new_lshape,
                                   &new_rshape,
                                   &new_oshape);
  if (ndim == 0) {
    return false;
  }
  int total_size = 1;
  for (auto dim : new_oshape) {
    total_size *= dim;
  }
  DEEPNET_BROADCAST_NDIM_SWITCH(ndim, NDim, {
    Shape<NDim> oshape(new_oshape);
    Shape<NDim> lstride = calc_stride<NDim>(new_lshape);
    Shape<NDim> rstride = calc_stride<NDim>(new_rshape);
    KernelLauncher<broadcast_kernel<NDim, DType, OP>, Context>::Launch(total_size,
                                                               context,
                                                               lstride,
                                                               rstride,
                                                               oshape,
                                                               input1,
                                                               input2,
                                                               output);
  });
  return true;
}

template<typename IType, typename DType, typename OP, template<typename, typename> class KernelLauncher,
    class Context>
bool BroadcastCompute(const IType* condition,
                      const std::vector<TIndex>& condition_shape,
                      const DType* input1,
                      const std::vector<TIndex>& shape1,
                      const DType* input2,
                      const std::vector<TIndex>& shape2,
                      DType* output,
                      const std::vector<TIndex>& outshape,
                      const Context& context) {
  std::vector<TIndex> new_condition_shape, new_lshape, new_rshape, new_oshape;
  int ndim = BroadcastShapeCompact(condition_shape,
                                   shape1,
                                   shape2,
                                   outshape,
                                   &new_condition_shape,
                                   &new_lshape,
                                   &new_rshape,
                                   &new_oshape);
  if (ndim == 0) {
    return false;
  }
  int total_size = 1;
  for (auto dim : new_oshape) {
    total_size *= dim;
  }
  DEEPNET_BROADCAST_NDIM_SWITCH(ndim, NDim, {
    Shape<NDim> oshape(new_oshape);
    Shape<NDim> cstride = calc_stride<NDim>(new_condition_shape);
    Shape<NDim> lstride = calc_stride<NDim>(new_lshape);
    Shape<NDim> rstride = calc_stride<NDim>(new_rshape);
    KernelLauncher<broadcast_condition_kernel<NDim, IType, DType, OP>, Context>::Launch(total_size,
                                                                                        context,
                                                                                        cstride,
                                                                                        lstride,
                                                                                        rstride,
                                                                                        oshape,
                                                                                        condition,
                                                                                        input1,
                                                                                        input2,
                                                                                        output);
  });
  return true;
}

} // namespace broadcast
} // namespace blaze
