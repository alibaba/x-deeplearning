/*
 * \file not_op.h
 * \desc The not operator
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct NotParam {
  DType* x;
  size_t size;
  DType* y;

  NotParam(DType* x, size_t size, DType* y) :
      x(x), size(size), y(y) { }
};

template <class Context>
class NotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  NotOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

}  // namespace blaze
