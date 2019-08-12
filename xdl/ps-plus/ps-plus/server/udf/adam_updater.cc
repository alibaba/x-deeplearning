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

using std::vector;

class AdamUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double>, vector<double>, vector<double>, vector<bool> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& epsilons,
      const vector<double>& beta1s,
      const vector<double>& beta2s,
      const vector<bool>& lr_decays) const {
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != epsilons.size()
        || sslices.size() != beta1s.size() || sslices.size() != beta2s.size() || sslices.size() != lr_decays.size()) {
      return Status::ArgumentError("AdamUpdater: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      if (!slices.writable) {
        return Status::ArgumentError("slice is not writable");
      }
      double learning_rate = learning_rates[si];
      double epsilon = epsilons[si];
      double beta1 = beta1s[si];
      double beta2 = beta2s[si];
      const Tensor& grad_tensor = grad_tensors[si];
      bool lr_decay = lr_decays[si];
      Tensor* data_tensor = slices.variable->GetData();
      Tensor* beta1_tensor = slices.variable->GetAnyOneSlot("beta1", DataType::kDouble, ps::TensorShape({}), [&]{ return new initializer::ConstantInitializer(beta1); });
      Tensor* beta2_tensor = slices.variable->GetAnyOneSlot("beta2", DataType::kDouble, ps::TensorShape({}), [&]{ return new initializer::ConstantInitializer(beta2); });
      Tensor* m_tensor = slices.variable->GetVariableLikeSlot("m", DataType::kDouble, []{ return new initializer::ConstantInitializer(0); });
      Tensor* v_tensor = slices.variable->GetVariableLikeSlot("v", DataType::kDouble, []{ return new initializer::ConstantInitializer(0); });    
    
      if (grad_tensor.Type() != data_tensor->Type()) {
        return Status::ArgumentError("grad should has same datatype with variable");
      }

      CASES(data_tensor->Type(), do {
            double* beta1_power = beta1_tensor->Raw<double>();
            double* beta2_power = beta2_tensor->Raw<double>();
            double alpha;
            if (lr_decay) {
              alpha = learning_rate * sqrt(1 - *beta2_power) / (1 - *beta1_power);
            } else {
              alpha = learning_rate;
            }
            MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                  for (size_t i = r.begin; i < r.end; i++) {
                    T* grad = grad_tensor.Raw<T>(i);
                    size_t slice = slices.slice_id[i];
                    if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                      continue;
                    }
                    T* data = data_tensor->Raw<T>(slice);
                    double* m = m_tensor->Raw<double>(slice);
                    double* v = v_tensor->Raw<double>(slice);
                    for (size_t j = 0; j < slices.slice_size; j++) {
                      double grad_d = (double)(*grad);
                      *m += (grad_d - *m) * (1 - beta1);
                      *v += (grad_d * grad_d - *v) * (1 - beta2);
                      *data -= (alpha * *m) / (sqrt(*v) + epsilon);
                      data++;grad++;m++;v++;
                    }
                  }
                  return Status::Ok();
                });
            *beta1_power *= beta1;
            *beta2_power *= beta2;
          } while(0));
    }
    return Status::Ok();    
  }
};

SIMPLE_UDF_REGISTER(AdamUpdater, AdamUpdater);

}
}
}

