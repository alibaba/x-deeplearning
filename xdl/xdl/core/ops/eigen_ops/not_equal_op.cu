#include "xdl/core/ops/eigen_ops/binary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tlhs, typename Trhs, typename Tout>
struct NotEqualFunctor {
  Tout operator()(Tlhs lhs, Trhs rhs) const {
    return lhs != rhs;
  }
};

}

XDL_REGISTER_BOOL_BINARY_EIGEN_OP_SIMPLE(NotEqual, xdl::NotEqualFunctor)
XDL_REGISTER_BOOL_BINARY_EIGEN_OP(NotEqual, xdl::NotEqualFunctor, bool)
