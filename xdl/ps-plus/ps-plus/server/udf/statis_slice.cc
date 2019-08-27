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

#include <math.h>
#include <mutex>

#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"

namespace ps {
namespace server {
namespace udf {

class StatisSlice : public SimpleUdf<std::vector<Slices>, std::vector<Tensor>, Tensor, Tensor, Tensor,
                                     std::string, std::vector<TensorSlices>*> {
 public:
  /*
   *  S[0] = a[0]
   *  S[N] = a[N] + decay * S[N-1]
   *       = a[N] + decay*a[N-1] + decay^2*a[N-2] + ... + decay^(N-1)*a[1] + decay^N*a[0]
   *       = a[N] + decay^(N-n) * S[n]
   *
   *  N = gs / period
   *  data = click + data * pow(decay, N - updated_n)
   *  updated_n = N
   */
  virtual Status SimpleRun(UdfContext* ctx,
                           const std::vector<Slices>& slices,
                           const std::vector<Tensor>& clicks,
                           const Tensor& global_step,
                           const Tensor& statis_decay,
                           const Tensor& statis_decay_period,
                           const std::string& statis_type,
                           std::vector<TensorSlices>* result) const {
    int64_t gs = *global_step.Raw<int64_t>();
    float decay = (float) *statis_decay.Raw<double>();
    int64_t period = *statis_decay_period.Raw<int64_t>();
    int64_t N = gs / period;
    static const int64_t N_begin = N;
    result->resize(slices.size());  // number of variables
    for (size_t si = 0; si < slices.size(); ++si) {
      const Slices& slice = slices[si];
      Tensor* data_tensor = slice.variable->GetData();
      Tensor* updated_n_tensor = slice.variable->GetVariableLikeSlot(slice.variable->GetName() + "_UPDATED_N", ps::types::DataType::kInt64,
                                                                     []{ return new initializer::ConstantInitializer(N_begin); });
      Statis(slice, clicks[si], decay, N, updated_n_tensor, data_tensor);
      (*result)[si].slice_size = slice.slice_size;
      (*result)[si].slice_id = slice.slice_id;
      (*result)[si].dim_part = slice.dim_part;
      (*result)[si].tensor = *data_tensor;
    }
    return Status::Ok();
  }

 private:
  inline void Statis(const Slices& slice, const Tensor& click, float decay, int64_t N,
                     Tensor* updated_n_tensor, Tensor* data_tensor) const {
    int32_t* pclick = click.Raw<int32_t>();
    CASES(data_tensor->Type(), MultiThreadDo(slice.slice_id.size(), [&](const Range& r) {
      for (size_t i = r.begin; i < r.end; ++i) {
        size_t s = slice.slice_id[i];
        if (s == ps::HashMap::NOT_ADD_ID) continue;
        int64_t* updated_n = updated_n_tensor->Raw<int64_t>(s);
        T* data = data_tensor->Raw<T>(s);
        if (N > *updated_n) {
          *data *= powf(decay, N - *updated_n);
        } else if (N < *updated_n - 1) {
          printf("WARNING: statis_slice %s (slice_id=%lu) received N=%ld, but updated_n=%ld\n",
                 slice.variable->GetName().c_str(), s, N, *updated_n);
          continue;
        }
        *data += pclick[i];
        *updated_n = N;
      }
      return Status::Ok();
    }));
  }
};

SIMPLE_UDF_REGISTER(StatisSlice, StatisSlice);

}
}
}

