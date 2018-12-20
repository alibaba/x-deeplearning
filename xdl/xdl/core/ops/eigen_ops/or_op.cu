#include "xdl/core/ops/eigen_ops/binary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tlhs, typename Trhs, typename Tout>
struct OrFunctor {
  Tout operator()(Tlhs lhs, Trhs rhs) const {
    return lhs | rhs;
  }
};

template<typename Device>
struct OrFunctor<Device, bool, bool, bool> {
  bool operator()(bool lhs, bool rhs) const {
    return lhs || rhs;
  }
};

}

XDL_REGISTER_INT_CALC_BINARY_EIGEN_OP_SIMPLE(Or_, xdl::OrFunctor)
XDL_REGISTER_CALC_BINARY_EIGEN_OP(Or_, xdl::OrFunctor, bool)
