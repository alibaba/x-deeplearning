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
#include "ps-plus/common/logging.h"
#include "ps-plus/common/hashmap.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;

class MovingAverageUpdater : public SimpleUdf<vector<Slices>, vector<float>, vector<Tensor> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<float>& moments,
      const vector<Tensor>& value_tensors) const {
    if (sslices.size() != value_tensors.size() || sslices.size() != moments.size()) {
      return Status::ArgumentError("MovingAverageUpdater: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      if (!slices.writable) {
        return Status::ArgumentError("slice is not writable");
      }
      float moment = moments[si];
      const Tensor& value_tensor = value_tensors[si];
      Tensor* data_tensor = slices.variable->GetData();
      if (value_tensor.Type() != data_tensor->Type()) {
        return Status::ArgumentError("value should has same datatype with variable");
      }
      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                for (size_t i = r.begin; i < r.end; i++) {
                  int64_t slice = slices.slice_id[i];
                  if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                    continue;
                  }
                  T* data = data_tensor->Raw<T>(slice);
                  T* value = value_tensor.Raw<T>(i);
                  for (size_t j = 0; j < slices.slice_size; ++j) {
                    *data = moment * (*data) + (1.0 - moment) * (*value);
                    data++;value++;
                  }
                }
                return Status::Ok();}));
    }
    return Status::Ok();
  }          
};

SIMPLE_UDF_REGISTER(MovingAverageUpdater, MovingAverageUpdater);

}
}
}

