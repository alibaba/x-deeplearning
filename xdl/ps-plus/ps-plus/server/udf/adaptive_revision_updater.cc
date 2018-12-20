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

class AdaptiveRevisionUpdater : public SimpleUdf<Slices, Tensor, double, double, double, size_t, size_t> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const Tensor& grad_tensor,
      const double& learning_rate_,
      const double& initial_accumulator_value_,
      const double& max_revision_ratio_,
      const size_t& worker_cnt_,
      const size_t& worker_idx_) const {
	if (!slices.writable) {
      return Status::ArgumentError("slice is not writable");
    }

    double learning_rate = learning_rate_;
    double initial_accumulator_value = initial_accumulator_value_;
    double max_revision_ratio = max_revision_ratio_;
    size_t worker_cnt = worker_cnt_;
    size_t worker_idx = worker_idx_;

    Tensor* data_tensor = slices.variable->GetData();
    Tensor* g_tensor = slices.variable->GetVariableLikeSlot("g", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(0.0); });
    
    std::vector<size_t> g_worker_dims{worker_cnt};
    const std::vector<size_t>& var_dims = data_tensor->Shape().Dims();
    g_worker_dims.insert(g_worker_dims.end(), var_dims.begin(), var_dims.end());
    Tensor* g_old_tensor = slices.variable->GetAnyOneSlot("g_old", data_tensor->Type(), ps::TensorShape(g_worker_dims), [=]{ return new initializer::ConstantInitializer(0.0); });
    
    Tensor* z_tensor = slices.variable->GetVariableLikeSlot("z", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
    Tensor* z2_tensor = slices.variable->GetVariableLikeSlot("z2", data_tensor->Type(), [=]{ return new initializer::ConstantInitializer(initial_accumulator_value); });
    
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
      T* g_base_ptr = g_tensor->Raw<T>();
      T* g_old_base_ptr = g_old_tensor->Raw<T>();
      g_old_base_ptr += worker_idx * data_tensor->Shape().NumElements();
      T* z_base_ptr = z_tensor->Raw<T>();
      T* z2_base_ptr = z2_tensor->Raw<T>();
      T* grad = grad_tensor.Raw<T>();

      size_t num_vals = 0;
      for (size_t i = 0; i < slices.slice_id.size(); ++i) {
        num_vals += slices.slice_size;
      }
      T* delta2_base = new T[num_vals];
      T* delta2 = delta2_base;
      T delta1_sum = 0.0;
      T delta2_sum = 0.0;

      for (size_t slice : slices.slice_id) {
        if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
          continue;
        }          
        T* data = data_ptr + slice * slices.slice_size;
        T* g = g_base_ptr + slice * slices.slice_size;
        T* g_old = g_old_base_ptr + slice * slices.slice_size;
        T* z = z_base_ptr + slice * slices.slice_size;
        T* z2 = z2_base_ptr + slice * slices.slice_size;

        for (size_t i = 0; i < slices.slice_size; ++i) {
          T g_bck = *g - *g_old;
          T lr_old = learning_rate / sqrt(*z2);
          T z_delta = *grad * *grad + 2 * *grad * g_bck;
          T z_new = *z + z_delta;
          T z2_new = fmax(z_new, *z2);
          T z2_delta = z2_new - *z2;
          T lr = learning_rate / sqrt(z2_new);
          T delta1 = - lr * *grad;
          *delta2 = (lr_old - lr) * g_bck;
          delta1_sum += delta1 * delta1;
          delta2_sum += *delta2 * *delta2;

          *data += delta1;
          *g += *grad;
          *z += z_delta;
          *z2 += z2_delta;

          data++; g++; g_old++; z++; z2++; delta2++; grad++;
        }
      }
      // Stabilize the initial revision process.
      T delta1_dot = sqrt(delta1_sum);
      T delta2_dot = sqrt(delta2_sum);
      T delta_ratio = delta2_dot / delta1_dot;
      if (delta_ratio > max_revision_ratio) {
        for (size_t i = 0; i < num_vals; ++i) {
          delta2_base[i] = delta2_base[i] * max_revision_ratio / delta_ratio;
        }
      }

      delta2 = delta2_base;
      for (size_t slice : slices.slice_id) {
        T* data = data_ptr + slice * slices.slice_size;
        for (size_t i = 0; i < slices.slice_size; ++i) {
          *data += *delta2;
          data++;
          delta2++;
        }
      }

      delete[] delta2_base;
    } while(0));
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(AdaptiveRevisionUpdater, AdaptiveRevisionUpdater);

}
}
}

