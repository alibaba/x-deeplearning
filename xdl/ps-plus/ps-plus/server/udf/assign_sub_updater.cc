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

class AssignSubUpdater : public SimpleUdf<Slices, Tensor> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const Tensor& grad_tensor) const {
    if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    Tensor* data_tensor = slices.variable->GetData();
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
      for (size_t slice : slices.slice_id) {
          if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
            grad += slices.slice_size;
            continue;
          }          
          T* data = data_ptr + slice * slices.slice_size;
          for (size_t i = 0; i < slices.slice_size; i++) {
              *data -= *grad;
              data++;grad++;
          }
      }
    } while(0));
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AssignSubUpdater, AssignSubUpdater);

}
}
}

