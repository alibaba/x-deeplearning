/*
 * \file sigmoid_op.h
 * \desc The sigmoid operator
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct SigmoidParam {
  DType* x;
  size_t size;
  DType* y;

  SigmoidParam(DType* x, size_t size, DType* y) :
      x(x), size(size), y(y) { }
};

template <class Context>
class SigmoidOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  SigmoidOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

}  // namespace blaze
