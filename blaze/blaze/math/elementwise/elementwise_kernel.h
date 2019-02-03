/*
 * \file elementwise_kernel.h
 * \desc the base calculation kernel of elementwise op  
 */
#pragma once

#include "blaze/common/common_defines.h"

namespace blaze {
namespace broadcast {

struct Sum {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a + b;
  }
};

struct Sub {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a - b;
  }
};

struct Mul {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a * b;
  }
};

struct Div {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a / b;
  }
};

struct Equal {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a == b ? 1 : 0;
  }
};

struct NotEqual {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a == b ? 0 : 1;
  }
};

struct Assign {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a;
  }
};

struct Max {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a > b ? a : b;
  }
};

struct Min {
  template <typename DType>
  BLAZE_INLINE_X static DType Map(DType a, DType b) {
    return a > b ? b : a;
  }
};

struct Where {
  template<typename IType, typename DType>
  BLAZE_INLINE_X static DType Map(IType condition, DType a, DType b) {
    return condition > 0 ? a : b;
  }
};

} // namespace broadcast
} // namespace blaze
