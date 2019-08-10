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

class FtrlUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& learning_rate_powers,
      const vector<double>& initial_accumulator_values,
      const vector<double>& l1_regs,
      const vector<double>& l2_regs) const {
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != learning_rate_powers.size() || sslices.size() != initial_accumulator_values.size() || sslices.size() != l1_regs.size() || sslices.size() != l1_regs.size()) {
      return Status::ArgumentError("FtrlUpdater: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      if (!slices.writable) {
        return Status::ArgumentError("slice is not writable");
      }
      double learning_rate = learning_rates[si];
      double learning_rate_power = learning_rate_powers[si];
      double initial_accumulator_value = initial_accumulator_values[si];
      double l1_reg = l1_regs[si];
      double l2_reg = l2_regs[si];
      const Tensor& grad_tensor = grad_tensors[si];

      Tensor* data_tensor = slices.variable->GetData();
      Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accum", data_tensor->Type(), [&]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
      Tensor* linear_tensor = slices.variable->GetVariableLikeSlot("linear", data_tensor->Type(), [&]{ return new initializer::ConstantInitializer(0); });
      if (grad_tensor.Type() != data_tensor->Type()) {
        return Status::ArgumentError("grad should has same datatype with variable");
      }

      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                for (size_t i = r.begin; i < r.end; i++) {
                  T* grad = grad_tensor.Raw<T>(i);
                  int64_t slice = slices.slice_id[i];
                  if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                    continue;
                  }
                  T* data = data_tensor->Raw<T>(slice);
                  T* acc = acc_tensor->Raw<T>(slice);
                  T* linear = linear_tensor->Raw<T>(slice);
                  for (size_t j = 0; j < slices.slice_size; j++) {
                    T new_accum = *acc + *grad * *grad;
                    if (fabs(learning_rate_power + 0.5) < 1e-6) {
                      *linear += *grad - (sqrt(new_accum) - sqrt(*acc)) / learning_rate * *data;
                      auto x = l1_reg * sgn(*linear) - *linear;
                      auto y = sqrt(new_accum) / learning_rate + l2_reg * 2;
                      auto pre_shrink = x / y;
                      if (fabs(*linear) > l1_reg) {
                        *data = pre_shrink;
                      } else {
                        *data = 0;
                      }
                    } else {
                      *linear += *grad - (pow(new_accum, -learning_rate_power) - pow(*acc, -learning_rate_power)) / learning_rate * *data;
                      auto x = l1_reg * sgn(*linear) - *linear;
                      auto y = pow(new_accum, -learning_rate_power) / learning_rate + l2_reg * 2;
                      auto pre_shrink = x / y;
                      if (fabs(*linear) > l1_reg) {
                        *data = pre_shrink;
                      } else {
                        *data = 0;
                      }
                    }
                    *acc += *grad * *grad;
                    data++; grad++; acc++; linear++;
                  }
                }
                return Status::Ok();
              }));
    }
    return Status::Ok();;
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

