/*
 * \file tanh_op.h 
 * \brief The tanh operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename Context>
class TanhOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  TanhOp(const OperatorDef& def, Workspace* workspace) :
    Operator<Context>(def, workspace) { }
  bool RunOnDevice() override;
};

}  // namespace blaze

