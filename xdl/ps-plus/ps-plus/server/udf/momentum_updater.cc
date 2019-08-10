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

class MomentumUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double>, vector<bool> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& momentums,
      const vector<bool>& use_nesterovs) const {
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != momentums.size() || sslices.size() != use_nesterovs.size()) {
      return Status::ArgumentError("MomentumUpdater: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
            const Slices& slices = sslices[si];
            std::unique_ptr<QRWLocker> locker;
            locker.reset(new QRWLocker(slices.variable->VariableLock(), QRWLocker::kSimpleRead));
            if (!slices.writable) {
              return Status::ArgumentError("slice is not writable");
            }
            double learning_rate = learning_rates[si];
            double momentum = momentums[si];
            bool use_nesterov = use_nesterovs[si];
            const Tensor& grad_tensor = grad_tensors[si];
            Tensor* data_tensor = slices.variable->GetData();
            Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accumulation", data_tensor->Type(), []{ return new initializer::ConstantInitializer(0); });
            if (grad_tensor.Type() != data_tensor->Type()) {
              return Status::ArgumentError("grad should has same datatype with variable");
            }

            CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                      for (size_t i = r.begin; i < r.end; i++) {
                        int64_t slice = slices.slice_id[i];
                        if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                          continue;
                        }                  
                        T* data = data_tensor->Raw<T>(slice);
                        T* acc = acc_tensor->Raw<T>(slice);
                        T* grad = grad_tensor.Raw<T>(i);
                        if (use_nesterov) {
                          for (size_t j = 0; j < slices.slice_size; j++) {
                            *acc = *acc * momentum + *grad;
                            *data -= *grad * learning_rate + *acc * momentum * learning_rate;
                            data++; acc++; grad++;
                          }
                        } else {
                          for (size_t j = 0; j < slices.slice_size; j++) {
                            *acc = *acc * momentum + *grad;
                            *data -= *acc * learning_rate;
                            data++; acc++; grad++;
                          }
                        }
                      }
                      return Status::Ok();
                    }));
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(MomentumUpdater, MomentumUpdater);

}
}
}

