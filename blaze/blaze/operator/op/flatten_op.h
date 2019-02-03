/*
 * \file flatten_op.h 
 * \brief The flatten operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FlattenOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace, false) {
    for (const std::string& input_str : def.input()) {
      Blob* blob = workspace->CreateBlob(input_str, this->device_option_);
      CHECK(blob != nullptr, "op: %s has non-exitsing input blob",
            def.type().c_str(), input_str.c_str());
      this->inputs_.push_back(blob);
      this->outputs_.push_back(blob);  // should use ref-blob.
      workspace->SetBlob(def.output(0), blob);
    }
    axis_ = OperatorBase::GetSingleArgument<TIndex>("axis", 1);
  }

  bool RunOnDevice() override {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);
    const std::vector<TIndex>& shape = x->shape();
    TIndex s1 = 1, s2 = 1;
    for (TIndex k = 0; k < axis_; ++k) s1 *= shape[k];
    for (TIndex k = axis_; k < shape.size(); ++k) s2 *= shape[k];
    y->Reshape({ s1, s2 });
    return true;
  }

 protected:
  TIndex axis_;
};

}  // namespace blaze
