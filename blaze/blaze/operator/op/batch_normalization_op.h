/*
 * \file batch_normalization_op.h 
 * \brief The batch normalization operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct BatchNormalizationParam {
  DType* x;
  size_t size;
  DType* gamma;
  DType* beta;
  DType* mean;
  DType* var;
  size_t c;
  DType* y;
  bool nosqrt;
  float eps;

  BatchNormalizationParam(DType* x,
                          size_t size,
                          DType* gamma,
                          DType* beta,
                          DType* mean,
                          DType* var,
                          size_t c,
                          DType* y,
                          bool nosqrt,
                          float eps) :
      x(x),
      size(size),
      gamma(gamma),
      beta(beta),
      mean(mean),
      var(var),
      c(c),
      y(y),
      nosqrt(nosqrt),
      eps(eps) { }
};

template <class Context>
class BatchNormalizationOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  BatchNormalizationOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    nosqrt_ = OperatorBase::GetSingleArgument<bool>("nosqrt", false);    
    eps_ = OperatorBase::GetSingleArgument<float>("eps", kBnEpsilon);
  }

  bool RunOnDevice() override;

 protected:
  void CheckValid() {
    Blob* x = this->Input(0);
    Blob* gamma = this->Input(1);
    Blob* beta = this->Input(2);
    Blob* mean = this->Input(3);
    Blob* var = this->Input(4);

    // check data type
    int data_type[4] = { gamma->data_type(),
                         beta->data_type(),
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
    size_t size[4] = { gamma->size(),
                       beta->size(),
                       mean->size(),
                       var->size()
                      };
    for (size_t k = 1; k < 4; ++k) {
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
  }

  bool nosqrt_ = false;
  float eps_;
};

}  // namespace blaze

