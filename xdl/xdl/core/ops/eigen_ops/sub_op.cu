#include "xdl/core/ops/eigen_ops/binary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tlhs, typename Trhs, typename Tout>
struct SubFunctor {
  Tout operator()(Tlhs lhs, Trhs rhs) const {
    return lhs - rhs;
  }
};

}

XDL_REGISTER_CALC_BINARY_EIGEN_OP_SIMPLE(Sub, xdl::SubFunctor)
