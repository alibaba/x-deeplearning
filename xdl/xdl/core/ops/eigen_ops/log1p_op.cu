#include "xdl/core/ops/eigen_ops/unary_eigen_op_common.h"
#include <cmath>

namespace xdl {

template <typename Device, typename Tin, typename Tout>
struct Log1pFunctor {
  Tout operator()(Tin in) const {
    return log1p(in);
  }
};

}

XDL_REGISTER_FLOAT_CALC_UNARY_EIGEN_OP_SIMPLE(Log1p, xdl::Log1pFunctor)
