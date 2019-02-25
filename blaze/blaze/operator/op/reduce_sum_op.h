/*
 * \file reduce_sum_op.h 
 * \brief The reduce sum operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/gemm.h"
#include "blaze/math/vml.h"
#include "blaze/math/reduce.h"

namespace blaze {

template <class Context>
class ReduceSumOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  ReduceSumOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    if (OperatorBase::HasArgument("axes")) {
      std::vector<size_t> axes = OperatorBase::GetRepeatedArgument<size_t>("axes");
      BLAZE_CONDITION_THROW(axes.size() == 1, "axes.size()=", axes.size());
      axis_ = axes[0];
    } else {
      axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 0);
    }
    keepdims_ = OperatorBase::GetSingleArgument<int>("keepdims", 1);
  }

  bool RunOnDevice() override {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);
    
    CheckValid();

    const std::vector<TIndex>& shape = x->shape();
    std::vector<TIndex> new_shape;
    size_t batch_count = 1;
    size_t K = 1, N = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == axis_) {
        if (keepdims_) new_shape.push_back(1);
      } else new_shape.push_back(shape[i]);
      
      if (i < axis_) batch_count *= shape[i];
      else if (i == axis_) K = shape[i];
      else N *= shape[i];
    }
    y->Reshape(new_shape);

    TYPE_SWITCH_WITH_CTX(this->context_, x->data_type(), DType, {
    ReduceSum<DType, Context>(x->as<DType>(),
                              batch_count,
                              K,
                              N,
                              y->as<DType>(),
                              &this->context_);            
    });
    return true;
  }

 protected:
  void CheckValid() {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);

    const std::vector<TIndex>& shape = x->shape();
    BLAZE_CONDITION_THROW(axis_ < shape.size(),
                          "axis_=",
                          axis_,
                          " shape.size()=",
                          shape.size(), " ", this->def_.name());
    BLAZE_CONDITION_THROW(x->data_type() == y->data_type(),
                          "x->data_type()=",
                          x->data_type(),
                          " y->data_type()=",
                          y->data_type());
  }

  size_t axis_;
  int keepdims_;
};

}  // namespace blaze

