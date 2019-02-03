/*
 * \file cast_op.h 
 * \brief The cast operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename SrcDType, typename DstDType>
struct CastParam {
  SrcDType* x;
  size_t size;
  DstDType* y;

  CastParam(SrcDType* x, size_t size, DstDType* y) : x(x), size(size), y(y) { }
};

template <class Context>
class CastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  CastOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    to_ = OperatorBase::GetSingleArgument<int>("to", kFloat);    
  }

  bool RunOnDevice() override;

 protected:
  int to_;
};

}  // namespace blaze
