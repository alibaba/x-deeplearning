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

class FtrlUpdater : public SimpleUdf<Slices, Tensor, double, double, double, double, double> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const Tensor& grad_tensor,
      const double& learning_rate_,
      const double& learning_rate_power_,
      const double& initial_accumulator_value_,
      const double& l1_reg_,
      const double& l2_reg_) const {
    if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    double learning_rate = learning_rate_;
    double learning_rate_power = learning_rate_power_;
    double initial_accumulator_value = initial_accumulator_value_;
    double l1_reg = l1_reg_;
    double l2_reg = l2_reg_;

    Tensor* data_tensor = slices.variable->GetData();
    Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accum", data_tensor->Type(), [&]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
    Tensor* linear_tensor = slices.variable->GetVariableLikeSlot("linear", data_tensor->Type(), [&]{ return new initializer::ConstantInitializer(0); });
    if (grad_tensor.Type() != data_tensor->Type()) {
      return Status::ArgumentError("grad should has same datatype with variable");
    }
    /*
    if (grad_tensor.Shape().NumElements() != slices.slice_size * slices.slice_id.size()) {
      return Status::ArgumentError("grad should has shape: " + std::to_string(slices.slice_size * slices.slice_id.size()));
    }
    */
    CASES(data_tensor->Type(), do {
      T* data_ptr = data_tensor->Raw<T>();
      T* grad = grad_tensor.Raw<T>();
      T* acc_ptr = acc_tensor->Raw<T>();
      T* linear_ptr = linear_tensor->Raw<T>();      

      for (size_t slice : slices.slice_id) {
          if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
            grad += slices.slice_size;
            continue;
          }            
          T* data = data_ptr + slice * slices.slice_size;
          T* accum = acc_ptr + slice * slices.slice_size;
          T* linear = linear_ptr + slice * slices.slice_size;
          for (size_t i = 0; i < slices.slice_size; i++) {
              T new_accum = *accum + *grad * *grad;
              if (fabs(learning_rate_power + 0.5) < 1e-6) {
                  *linear += *grad - (sqrt(new_accum) - sqrt(*accum)) / learning_rate * *data;
                  auto x = l1_reg * sgn(*linear) - *linear;
                  auto y = sqrt(new_accum) / learning_rate + l2_reg * 2;
                  auto pre_shrink = x / y;
                  if (fabs(*linear) > l1_reg) {
                      *data = pre_shrink;
                  } else {
                      *data = 0;
                  }
              } else {
                  *linear += *grad - (pow(new_accum, -learning_rate_power) - pow(*accum, -learning_rate_power)) / learning_rate * *data;
                  auto x = l1_reg * sgn(*linear) - *linear;
                  auto y = pow(new_accum, -learning_rate_power) / learning_rate + l2_reg * 2;
                  auto pre_shrink = x / y;
                  if (fabs(*linear) > l1_reg) {
                      *data = pre_shrink;
                  } else {
                      *data = 0;
                  }
              }
              *accum += *grad * *grad;
              data++; grad++; accum++; linear++;
          }
      }
    } while(0));
    return Status::Ok();
  }

protected:
    template <typename T>
    inline T sgn(const T x) const {
        return (T(0) < x) - (x < T(0));
    }
};

SIMPLE_UDF_REGISTER(FtrlUpdater, FtrlUpdater);

}
}
}

