/*
 * \file fuse_op.h 
 * \brief The fuse operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct FuseParam {
  DType* x1;
  size_t x1_m;
  size_t x1_n;
  DType* x2;
  size_t x2_m;
  size_t x2_n;
  DType* y;
  size_t y_m;
  size_t y_n;
};

template <class Context>
class FuseOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FuseOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    comm_index_ = OperatorBase::GetSingleArgument<int>("comm_index", 0);
  }

  bool RunOnDevice() override;

 protected:
  void CheckValid() {
    BLAZE_CONDITION_THROW(comm_index_ >= 0 && comm_index_ <= 1,
                          "comm_index_=", comm_index_);
    for (int i = 0; i < this->InputSize(); ++i) {
      Blob* x = this->Input(i);
      BLAZE_CONDITION_THROW(x->shape().size() == 2,
                            "x->shape().size()=", x->shape().size(),
                            " i=", i);
      if (comm_index_ == i) {
        // NOTE: We don't support batching in ulf ops.
        BLAZE_CONDITION_THROW(x->shape()[0] == 1, "x->shape()[0]=", x->shape()[0],
                              " i=", i, " ", this->def_.DebugString());
      } 
    }
  }

  template <typename DType>
  void Setup(FuseParam<DType>* params) {
    Blob* x1 = this->Input(0);
    Blob* x2 = this->Input(1);
    Blob* y = this->Output(0);
    
    params->x1 = x1->as<DType>();
    params->x1_m = x1->shape()[0];
    params->x1_n = x1->shape()[1];

    params->x2 = x2->as<DType>();
    params->x2_m = x2->shape()[0];
    params->x2_n = x2->shape()[1];

    params->y_m = std::max(params->x1_m, params->x2_m);
    params->y_n = params->x1_n + params->x2_n;

    std::vector<TIndex> shape;
    shape.push_back(params->y_m);
    shape.push_back(params->y_n);
    y->Reshape(shape);

    params->y = y->as<DType>();
  }

  int comm_index_;
};

}  // namespace blaze
