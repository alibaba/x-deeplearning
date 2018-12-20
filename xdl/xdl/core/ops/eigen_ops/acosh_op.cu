#include "xdl/core/ops/eigen_ops/unary_eigen_op_common.h"
#include <cmath>

namespace xdl {

template <typename Device, typename Tin, typename Tout>
struct AcoshFunctor {
  Tout operator()(Tin in) const {
    return acosh(in);
  }
};

}

XDL_REGISTER_FLOAT_CALC_UNARY_EIGEN_OP_SIMPLE(Acosh, xdl::AcoshFunctor)
