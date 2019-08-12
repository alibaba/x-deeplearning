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
   *       = decay^N * sum<j=0:N+1> { decay^(-j) * a[j] }
   *
   *  N = gs / period
   *  K = decay^(N-offset)
   *  acc = sum<j=0:N+1> { decay^(offset-j) * a[j] }
   *  data = K * acc
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
    static int64_t N_begin = N;
    static int64_t step = int64_t(logf(1e-30) / logf(decay));
    static int64_t offset = (N_begin / step) * step;
    static std::mutex mutex;
    if (N - offset > step) {
      std::lock_guard<std::mutex> lock(mutex);
      while (N - offset > step) {
        offset += step;
        UpdateOffset(slices, decay, step);
      }
    }
    float K = powf(decay, N - offset);
    result->resize(slices.size());  // number of variables
    for (size_t si = 0; si < slices.size(); ++si) {
      const Slices& slice = slices[si];
      Tensor* data_tensor = slice.variable->GetData();
      Tensor* acc_tensor = slice.variable->GetVariableLikeSlot(slice.variable->GetName() + "_ACC", data_tensor->Type(),
                                                               []{ return new initializer::ConstantInitializer(0); });
      Statis(slice, clicks[si], decay, offset, N, K, acc_tensor, data_tensor);
      (*result)[si].slice_size = slice.slice_size;
      (*result)[si].slice_id = slice.slice_id;
      (*result)[si].dim_part = slice.dim_part;
      (*result)[si].tensor = *data_tensor;
    }
    return Status::Ok();
  }

 private:
  inline void Statis(const Slices& slice, const Tensor& click, float decay, int64_t offset, int64_t N, float K,
                     Tensor* acc_tensor, Tensor* data_tensor) const {
    int32_t* pclick = click.Raw<int32_t>();
    CASES(data_tensor->Type(), MultiThreadDo(slice.slice_id.size(), [&](const Range& r) {
      for (size_t i = r.begin; i < r.end; ++i) {
        size_t s = slice.slice_id[i];
        if (s == ps::HashMap::NOT_ADD_ID) continue;
        T* acc = acc_tensor->Raw<T>(s);
        T* data = data_tensor->Raw<T>(s);
        *acc += pclick[i] * powf(decay, offset - N);
        *data = K * (*acc);
      }
      return Status::Ok();
    }));
  }

  inline void UpdateOffset(const std::vector<Slices>& slices, float decay, int64_t step) const {
    for (size_t si = 0; si < slices.size(); ++si) {
      const Slices& slice = slices[si];
      Tensor* data_tensor = slice.variable->GetData();
      Tensor* acc_tensor = slice.variable->GetVariableLikeSlot(slice.variable->GetName() + "_ACC", data_tensor->Type(),
                                                               []{ return new initializer::ConstantInitializer(0); });
      CASES(data_tensor->Type(), MultiThreadDo(acc_tensor->Shape().NumElements(), [&](const Range& r) {
        for (size_t i = r.begin; i < r.end; ++i) {
          T* acc = acc_tensor->Raw<T>(i);
          *acc *= powf(decay, step);
        }
        return Status::Ok();
      }));
    }
  }
};

SIMPLE_UDF_REGISTER(StatisSlice, StatisSlice);

}
}
}

