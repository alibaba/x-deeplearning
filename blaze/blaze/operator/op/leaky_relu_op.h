/*
 * \file leaky_relu_op.h
 * \desc The leaky relu operator.
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct LeakyReluParam {
  DType* x;
  size_t size;
  DType* y;
  float alpha;

  LeakyReluParam(DType* x, size_t size, DType* y, float alpha) :
      x(x), size(size), y(y), alpha(alpha) { }
};

template <class Context>
class LeakyReluOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  LeakyReluOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    alpha_ = OperatorBase::GetSingleArgument<float>("alpha", 0.01);
  }

  bool RunOnDevice() override;

 protected:
  float alpha_;
};

}  // namespace blaze

