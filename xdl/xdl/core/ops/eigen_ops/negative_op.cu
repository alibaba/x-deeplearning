#include "xdl/core/ops/eigen_ops/unary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tin, typename Tout>
struct NegativeFunctor {
  Tout operator()(Tin in) const {
    return -in;
  }
};

}

XDL_REGISTER_CALC_UNARY_EIGEN_OP_SIMPLE(Negative, xdl::NegativeFunctor)
