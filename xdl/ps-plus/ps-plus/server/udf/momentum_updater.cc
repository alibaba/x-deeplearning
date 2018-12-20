/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"

namespace ps {
namespace server {
namespace udf {

class MomentumUpdater : public SimpleUdf<Slices, Tensor, double, double, bool> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const Tensor& grad_tensor,
      const double& learning_rate_,
      const double& momentum_,
      const bool& use_nesterov_) const {
    if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    double learning_rate = learning_rate_;
    double momentum = momentum_;
    bool use_nesterov = use_nesterov_;
    Tensor* data_tensor = slices.variable->GetData();
    Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accumulation", data_tensor->Type(), []{ return new initializer::ConstantInitializer(0); });
    if (grad_tensor.Type() != data_tensor->Type()) {
      return Status::ArgumentError("grad should has same datatype with variable");
    }
    /*
    if (grad_tensor.Shape().NumElements() != slices.slice_size * slices.slice_id.size()) {
      return Status::ArgumentError("grad should has shape: " + std::to_string(slices.slice_size * slices.slice_id.size()));
    }
    */
    CASES(data_tensor->Type(), {
      T* data_ptr = data_tensor->Raw<T>();
      T* acc_ptr = acc_tensor->Raw<T>();
      T* grad = grad_tensor.Raw<T>();
      if (use_nesterov) {
        for (size_t slice : slices.slice_id) {
          if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {            
            grad += slices.slice_size;
            continue;
          }
          T* data = data_ptr + slice * slices.slice_size;
          T* acc = acc_ptr + slice * slices.slice_size;
          for (size_t i = 0; i < slices.slice_size; i++) {
            *acc = *acc * momentum + *grad;
            *data -= *grad * learning_rate + *acc * momentum * learning_rate;
            data++; acc++; grad++;
          }
        }
      } else {
        for (size_t slice : slices.slice_id) {
          if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
            grad += slices.slice_size;
            continue;
          }            
          T* data = data_ptr + slice * slices.slice_size;
          T* acc = acc_ptr + slice * slices.slice_size;
          for (size_t i = 0; i < slices.slice_size; i++) {
            *acc = *acc * momentum + *grad;
            *data -= *acc * learning_rate;
            data++; acc++; grad++;
          }
        }
      }
    });
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(MomentumUpdater, MomentumUpdater);

}
}
}

