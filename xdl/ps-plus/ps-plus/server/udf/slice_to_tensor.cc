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
#include "ps-plus/common/hashmap.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;

class SliceToTensor : public SimpleUdf<vector<Slices>, vector<Tensor>*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const vector<Slices>& sslices, vector<Tensor>* results) const {
    static char zero_buffer[1<<16] = {0};
    results->resize(sslices.size());
    std::promise<bool> ok;
    Status status = Status::Ok();    
    std::atomic<size_t> counter(sslices.size());
    for (size_t si = 0; si < sslices.size(); si++) {
      ThreadPool::Global()->Schedule([&, si]{
            const Slices& slices = sslices[si];
            const Tensor& t = *(slices.variable->GetData());
            if (slices.dim_part < 0) {
              (*results)[si] = t;
            } else {
              if ((size_t)slices.dim_part > t.Shape().Size()) {
                status = Status::ArgumentError("Slice dim_part Error");
                CHECK_COUNTER(counter, ok);
              }
              std::vector<size_t> dims(1, slices.slice_id.size());
              for (size_t i = slices.dim_part; i < t.Shape().Dims().size(); i++) {
                dims.push_back(t.Shape()[i]);
              }
              Tensor result(t.Type(), TensorShape{dims}, t.GetInitializer()->Clone(), ps::Tensor::TType::kContinuous, false);
              CASES(t.Type(), {
                    size_t chunk_size = slices.slice_size * sizeof(T);
                    for (size_t i = 0; i < slices.slice_id.size(); ++i) {
                      int64_t slice = slices.slice_id[i];
                      if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                        memcpy(result.Raw<T>(i), zero_buffer, chunk_size);
                      } else {
                        memcpy(result.Raw<T>(i), t.Raw<T>(slice), chunk_size);
                      }
                    }});
              (*results)[si] = result;
            }
            CHECK_COUNTER(counter, ok);            
          });
    }
    ok.get_future().wait();
    return status;
  }
};

SIMPLE_UDF_REGISTER(SliceToTensor, SliceToTensor);

}
}
}

