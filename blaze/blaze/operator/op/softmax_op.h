/*
 * \file softmax_op.h 
 * \brief The softmax operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename Context>
class SoftmaxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  SoftmaxOp(const OperatorDef& def, Workspace* workspace);
  bool RunOnDevice() override;

 protected:
  size_t axis_;

#ifdef USE_CUDA
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  std::shared_ptr<Blob> x_fp32_, y_fp32_;
#endif

  std::shared_ptr<Blob> iblob_;
};

}  // namespace blaze
