#include "xdl/core/ops/eigen_ops/unary_eigen_op_common.h"
#include <cmath>

namespace xdl {

template <typename Device, typename Tin, typename Tout>
struct SinhFunctor {
  Tout operator()(Tin in) const {
    return sinh(in);
  }
};

}

XDL_REGISTER_FLOAT_CALC_UNARY_EIGEN_OP_SIMPLE(Sinh, xdl::SinhFunctor)
