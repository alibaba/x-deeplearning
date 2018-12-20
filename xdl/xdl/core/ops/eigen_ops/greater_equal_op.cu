#include "xdl/core/ops/eigen_ops/binary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tlhs, typename Trhs, typename Tout>
struct GreaterEqualFunctor {
  Tout operator()(Tlhs lhs, Trhs rhs) const {
    return lhs >= rhs;
  }
};

}

XDL_REGISTER_BOOL_BINARY_EIGEN_OP_SIMPLE(GreaterEqual, xdl::GreaterEqualFunctor)
