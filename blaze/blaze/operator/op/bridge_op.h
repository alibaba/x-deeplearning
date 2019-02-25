/*
 * \file bridge_op.h 
 * \brief The bridge operation for copying data from cpu to gpu 
 */
#pragma once

#include "blaze/operator/operator.h"

namespace blaze {

template <class Context>
class BridgeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  
  BridgeOp(const OperatorDef& def, Workspace* workspace):
      Operator<Context>(def, workspace) {
    
  }
  
  bool RunOnDevice() override;

 protected:
  void CheckValid() {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);
    BLAZE_CONDITION_THROW(x->device_type() == kCPU,
        "x->device_type()=",
        x->device_type(),
        " != kCPU");
    BLAZE_CONDITION_THROW(y->device_type() == kCUDA,
        "y->device_type()=",
        y->device_type(),
        " != kCUDA");
    BLAZE_CONDITION_THROW(x->data_type() == y->data_type(),
        "x->data_type()=",
        x->data_type(),
        " != y->data_type()=",
        y->data_type());
  }

  void Setup() {
    const std::vector<size_t>& x_shape = this->Input(0)->shape();
    Blob* y = this->Output(0);
    y->Reshape(x_shape);
  }
};

} // namespace blaze
