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

class AdaptiveRevisionPullUpdater : public SimpleUdf<Slices, size_t, size_t> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const size_t& worker_cnt_,
      const size_t& worker_idx_) const {
	if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    size_t worker_cnt = worker_cnt_;
    size_t worker_idx = worker_idx_;

    Tensor* data_tensor = slices.variable->GetData();
    Tensor* g_tensor = slices.variable->GetVariableLikeSlot("g", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(0.0); });
    
    std::vector<size_t> g_worker_dims{worker_cnt};
    const std::vector<size_t>& var_dims = data_tensor->Shape().Dims();
    g_worker_dims.insert(g_worker_dims.end(), var_dims.begin(), var_dims.end());
    Tensor* g_old_tensor = slices.variable->GetAnyOneSlot("g_old", data_tensor->Type(), ps::TensorShape(g_worker_dims), [=]{ return new initializer::ConstantInitializer(0.0); });
    
    CASES(data_tensor->Type(), do {
      T* g_base_ptr = g_tensor->Raw<T>();
      T* g_old_base_ptr = g_old_tensor->Raw<T>();
      g_old_base_ptr += worker_idx * data_tensor->Shape().NumElements();

      for (auto slice : slices.slice_id) {
        if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
          continue;
        }          
        T* g = g_base_ptr + slice * slices.slice_size;
        T* g_old = g_old_base_ptr + slice * slices.slice_size;
        memcpy(g_old, g, slices.slice_size * sizeof(T));
      }
    } while(0));
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdaptiveRevisionPullUpdater, AdaptiveRevisionPullUpdater);

}
}
}

