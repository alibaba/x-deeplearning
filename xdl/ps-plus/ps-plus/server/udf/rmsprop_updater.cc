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

class RmspropUpdater : public SimpleUdf<vector<Slices>, vector<Tensor>, vector<double>, vector<double>, vector<double>, vector<double> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<Tensor>& grad_tensors,
      const vector<double>& learning_rates,
      const vector<double>& decays,
      const vector<double>& alphas, /* It's called 'momentum' in tf.RMSPropOptimizer, but it is actually a coefficiency. Thus we call it 'alpha'. */
      const vector<double>& epsilons) const {
    if (sslices.size() != grad_tensors.size() || sslices.size() != learning_rates.size() || sslices.size() != decays.size() || sslices.size() != alphas.size() || sslices.size() != epsilons.size()) {
      return Status::ArgumentError("Rmspropoptimizer: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      if (!slices.writable) {
        return Status::ArgumentError("slice is not writable");
      }
      Tensor* data_tensor = slices.variable->GetData();
      Tensor* acc_tensor = slices.variable->GetVariableLikeSlot("accumulator", data_tensor->Type(), []{ return new initializer::ConstantInitializer(0); });
      Tensor* mom_tensor = slices.variable->GetVariableLikeSlot("momentum", data_tensor->Type(), []{ return new initializer::ConstantInitializer(0); });
      double learning_rate = learning_rates[si];
      double decay = decays[si];
      double alpha = alphas[si];
      double epsilon = epsilons[si];
      const Tensor& grad_tensor = grad_tensors[si];
    
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
                  T* mom = mom_tensor->Raw<T>(slice);
                  T* grad = grad_tensor.Raw<T>(i);
                  for (size_t j = 0; j < slices.slice_size; j++) {
                    *acc += (*grad * *grad - *acc) * (1 - decay);
                    *mom = *mom * alpha + (*grad * learning_rate) / sqrt(*acc + epsilon);
                    *data -= *mom;
                    data++; grad++; acc++; mom++;
                  }
                }
                return Status::Ok();
              }));
    }
    return Status::Ok(); 
  }
};

SIMPLE_UDF_REGISTER(RmspropUpdater, RmspropUpdater);

}
}
}

