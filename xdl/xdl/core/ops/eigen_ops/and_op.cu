#include "xdl/core/ops/eigen_ops/binary_eigen_op_common.h"

namespace xdl {

template <typename Device, typename Tlhs, typename Trhs, typename Tout>
struct AndFunctor {
  Tout operator()(Tlhs lhs, Trhs rhs) const {
    return lhs & rhs;
  }
};

template<typename Device>
struct AndFunctor<Device, bool, bool, bool> {
  bool operator()(bool lhs, bool rhs) const {
    return lhs && rhs;
  }
};

}

XDL_REGISTER_INT_CALC_BINARY_EIGEN_OP_SIMPLE(And_, xdl::AndFunctor)
XDL_REGISTER_CALC_BINARY_EIGEN_OP(And_, xdl::AndFunctor, bool)
