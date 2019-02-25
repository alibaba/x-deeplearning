/*
 * \file dice_op.h 
 * \brief The batch normalization operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct DiceParam {
  DType* x;
  size_t size;
  DType* gamma;
  DType* mean;
  DType* var;
  size_t c;
  DType* y;
  bool nosqrt;

  DiceParam(DType* x, size_t size,
            DType* gamma,
            DType* mean,
            DType* var,
            size_t c,
            DType* y,
            bool nosqrt) :
      x(x),
      size(size),
      gamma(gamma),
      mean(mean),
      var(var),
      c(c),
      y(y),
      nosqrt(nosqrt) { }
};

template <class Context>
class DiceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  DiceOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    nosqrt_ = OperatorBase::GetSingleArgument<bool>("nosqrt", false);    
  }

  bool RunOnDevice() override;

 protected:
  void CheckValid() {
    Blob* x = this->Input(0);
    Blob* gamma = this->Input(1);
    Blob* mean = this->Input(2);
    Blob* var = this->Input(3);

    // check data type
    int data_type[4] = { x->data_type(),
                         gamma->data_type(),
                         mean->data_type(),
                         var->data_type(),
                       };
    for (size_t k = 1; k < 4; ++k) {
      BLAZE_CONDITION_THROW(data_type[k] == data_type[k - 1],
                            "data_type[",
                            k,
                            "]=",
                            data_type[k],
                            " data_type[",
                            k - 1,
                            "]=",
                            data_type[k - 1]);
    }
    // check size.
    size_t size[3] = { gamma->size(),
                       mean->size(),
                       var->size()
                      };
    for (size_t k = 1; k < 3; ++k) {
      BLAZE_CONDITION_THROW(size[k] == size[k - 1],
                            "size[",
                            k,
                            "]=",
                            size[k],
                            "size[",
                            k - 1,
                            "]=",
                            size[k - 1]);
    }
    BLAZE_CONDITION_THROW(x->size() % gamma->size() == 0, "x->size()=", x->size(),
                          " gamma->size()=", gamma->size());
  }

  bool nosqrt_ = false;
};

}  // namespace blaze

