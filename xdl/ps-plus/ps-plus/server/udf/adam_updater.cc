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

class AdamUpdater : public SimpleUdf<Slices, Tensor, double, double, double, double, bool> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const Tensor& grad_tensor,
      const double& learning_rate_,
      const double& epsilon_,
      const double& beta1_,
      const double& beta2_,
      const bool& lr_decay_) const {
    if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    double learning_rate = learning_rate_;
    double epsilon = epsilon_;
    double beta1 = beta1_;
    double beta2 = beta2_;
    bool lr_decay = lr_decay_;
    Tensor* data_tensor = slices.variable->GetData();
    
    Tensor* beta1_tensor = slices.variable->GetAnyOneSlot("beta1", DataType::kDouble, ps::TensorShape({}), [&]{ return new initializer::ConstantInitializer(beta1); });
    Tensor* beta2_tensor = slices.variable->GetAnyOneSlot("beta2", DataType::kDouble, ps::TensorShape({}), [&]{ return new initializer::ConstantInitializer(beta2); });
    Tensor* m_tensor = slices.variable->GetVariableLikeSlot("m", DataType::kDouble, []{ return new initializer::ConstantInitializer(0); });
    Tensor* v_tensor = slices.variable->GetVariableLikeSlot("v", DataType::kDouble, []{ return new initializer::ConstantInitializer(0); });    
    
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
      double* beta1_power = beta1_tensor->Raw<double>();
      double* beta2_power = beta2_tensor->Raw<double>();
      double* m_ptr = m_tensor->Raw<double>();
      double* v_ptr = v_tensor->Raw<double>();

      double alpha;
      if (lr_decay) {
        alpha = learning_rate * sqrt(1 - *beta2_power) / (1 - *beta1_power);
      } else {
        alpha = learning_rate;
      }

      for (size_t slice : slices.slice_id) {
        if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
          grad += slices.slice_size;
          continue;
        }          
        T* data = data_ptr + slice * slices.slice_size;
        double* m = m_ptr + slice * slices.slice_size;
        double* v = v_ptr + slice * slices.slice_size;
        for (size_t i = 0; i < slices.slice_size; i++) {
          double grad_d = (double)(*grad);

          *m += (grad_d - *m) * (1 - beta1);
          *v += (grad_d * grad_d - *v) * (1 - beta2);
          *data -= (alpha * *m) / (sqrt(*v) + epsilon);
          data++;grad++;m++;v++;
        }
      }
      *beta1_power *= beta1;
      *beta2_power *= beta2;
    } while(0));
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdamUpdater, AdamUpdater);

}
}
}

