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
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/server/streaming_model_utils.h"
#include <iostream>
#include <map>

namespace ps {
namespace server {
namespace udf {

class AggregateSlice : public SimpleUdf<Slices, int64_t, int, Tensor, Slices*, Tensor*> {
 public:
   virtual Status SimpleRun(UdfContext* ctx, const Slices& input_slices, const int64_t& token, const int& worker_count, const Tensor& input_grad, Slices* output_slices, Tensor* output_grad) const {
       std::vector<Tensor> grads = {};
       std::vector<Slices> slices = {};
       Tensor tensor_cp(input_grad.Type(), input_grad.Shape(), new ps::initializer::ConstantInitializer(0.0), false);
       QuickMemcpy(tensor_cp.Raw<char>(), input_grad.Raw<char>(), SizeOfType(input_grad.Type()) * input_grad.Shape().NumElements());
       std::string variable_name = ctx->GetVariableName();
       lock_.lock();
       if (current_tokens[variable_name] != token) {
           current_tokens[variable_name] = token;
           grads_map[variable_name] = {};
           slices_map[variable_name] = {};
       }

       grads_map[variable_name].push_back(tensor_cp);
       slices_map[variable_name].push_back(input_slices);
       if (grads_map[variable_name].size() == (size_t) worker_count) {
           grads = grads_map[variable_name];
           slices = slices_map[variable_name];
           grads_map[variable_name] = {};
           slices_map[variable_name] = {};
       }
       lock_.unlock();
       
       std::map<size_t, size_t> id_count;
       for (const Slices& cur_slices: slices) {
           for (size_t id : cur_slices.slice_id) {
               auto iter = id_count.find(id);
               if (iter == id_count.end()) {
                   id_count[id] = 1;
               } else {
                   iter->second++;
               }
           }
       }
       std::unordered_map<size_t, size_t> id_position;
       size_t base_index = 0;
       for (auto iter = id_count.begin(); iter != id_count.end(); ++iter) {
           id_position[iter->first] = base_index++;
       } 
       size_t total_id = id_count.size();
       TensorShape new_shape = input_grad.Shape();
       if (new_shape.Size() > 1 && input_slices.dim_part != -1) {
           new_shape.Set(0, total_id);
       }
       *output_grad = Tensor(input_grad.Type(), new_shape, new ps::initializer::ConstantInitializer(0.0), true);
       for (size_t i = 0; i < grads.size(); i++) {
           const Slices& cur_slices = slices[i];
           const Tensor& cur_grad = grads[i];
           CASES(input_grad.Type(), do {
                       T* grad = cur_grad.Raw<T>();
                       T* out = output_grad->Raw<T>();
                       for (size_t j = 0; j < cur_slices.slice_id.size(); j++) {
                           size_t index = id_position[(cur_slices.slice_id[j])];
                           T* p = out + index * cur_slices.slice_size;
                           for (size_t k = 0; k < cur_slices.slice_size; k++) {
                               *p += *grad;
                               ++p;++grad;
                           }
                       }
                   } while(0));
       }

       output_slices->writable = input_slices.writable;
       output_slices->variable = input_slices.variable;
       output_slices->dim_part = input_slices.dim_part;
       output_slices->slice_size = input_slices.slice_size;
       CASES(input_grad.Type(), do {
                   T* out = output_grad->Raw<T>();
                   for (auto iter = id_count.begin(); iter != id_count.end(); ++iter) {
                       output_slices->slice_id.push_back(iter->first);
                       for (size_t i = 0; i < input_slices.slice_size; i++) {
                           *out++ /= (T)iter->second;
                       }
                   }
               } while(0));
       return Status::Ok();
   }
private:
    static std::mutex lock_;
    static std::map<std::string, std::vector<Slices>> slices_map;
    static std::map<std::string, std::vector<Tensor>> grads_map;
    static std::map<std::string, int64_t> current_tokens;
};

std::map<std::string, std::vector<Slices>> AggregateSlice::slices_map;
std::map<std::string, std::vector<Tensor>> AggregateSlice::grads_map;
std::map<std::string, int64_t> AggregateSlice::current_tokens;
std::mutex AggregateSlice::lock_;

SIMPLE_UDF_REGISTER(AggregateSlice, AggregateSlice);

}
}
}
