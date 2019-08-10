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
#include "ps-plus/common/logging.h"
#include "ps-plus/common/thread_pool.h"
#include <iostream>
#include <map>

namespace ps {
namespace server {
namespace udf {

using std::vector;

class AggregateSlice : public SimpleUdf<vector<Slices>, int64_t, int, vector<Tensor>, vector<Slices>*, vector<Tensor>*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const vector<Slices>& sslices, const int64_t& token, const int& worker_count, const vector<Tensor>& input_grads, vector<Slices>* output_slices, vector<Tensor>* output_grads) const {
    if (sslices.size() != input_grads.size()) {
      return Status::ArgumentError("AggregateSlice: slices and other size not match");
    }

    //for dense and sparse, token_name and variable_name are same, but for hash variable, token_name is fake.
    const std::string& token_name = ctx->GetVariableName();
    lock_.lock();
    if (current_tokens[token_name].first != token) {
      if (current_tokens[token_name].second != worker_count) {
        LOG(WARNING) << "AggregateSlice: receive wrong token [" << token << "] current token[" << current_tokens[token_name].first << "]";
      }
      current_tokens[token_name] = std::make_pair(token, 0);
      for (size_t si = 0; si < sslices.size(); si++) {
        const Slices& input_slices = sslices[si];
        std::string variable_name = input_slices.variable->GetName();
        grads_map[variable_name] = {};
        slices_map[variable_name] = {};
      }
    }
    current_tokens[token_name].second += 1;
    lock_.unlock();

    output_slices->resize(sslices.size());
    output_grads->resize(sslices.size());
    for (size_t si = 0; si < sslices.size(); si++) {
      const Tensor& input_grad = input_grads[si];
      const Slices& input_slices = sslices[si];
      const std::string& variable_name = input_slices.variable->GetName();

      lock_.lock();

      std::vector<Tensor> grads;
      std::vector<Slices> slices;

      grads_map[variable_name].push_back(input_grad.Clone());
      slices_map[variable_name].push_back(input_slices);
      if (current_tokens[token_name].second == worker_count) {
        grads = std::move(grads_map[variable_name]);
        slices = std::move(slices_map[variable_name]);
        grads_map[variable_name] = {};
        slices_map[variable_name] = {};
      }
      lock_.unlock();

      (*output_slices)[si].writable = input_slices.writable;
      (*output_slices)[si].variable = input_slices.variable;
      (*output_slices)[si].dim_part = input_slices.dim_part;
      (*output_slices)[si].slice_size = input_slices.slice_size;
      if (grads.size() == 0) {
        (*output_grads)[si] = Tensor(input_grad.Type(), TensorShape({}), new ps::initializer::ConstantInitializer(0.0));
        continue;
      }
      HashMap* id_position = new HashMapImpl<int64_t>(100);
      tbb::concurrent_vector<size_t> no_use;
      MultiThreadDoTBB(slices.size(), [&](const Range& r) {
            for (size_t i = r.begin; i < r.end; i++) {
              const Slices& cur_slices = slices[i];
              std::vector<size_t> position;
              size_t filter_count;
              size_t ret = id_position->Get((int64_t*)&cur_slices.slice_id[0], cur_slices.slice_id.size(), false, 1.0, &position, &no_use, &filter_count);
            }
            return Status::Ok();});
      size_t ids_size = id_position->GetSize();
      // empty slices
      if (ids_size == 0) {
        (*output_grads)[si] = Tensor(input_grad.Type(), TensorShape({}), new ps::initializer::ConstantInitializer(0.0));
        continue;
      }
      TensorShape output_shape = input_grad.Shape();
      if (output_shape.Size() > 1 && input_slices.dim_part != -1) {
        output_shape.Set(0, ids_size);
      }
      Tensor output_grad(input_grad.Type(), output_shape, new ps::initializer::ConstantInitializer(0.0));
      (*output_slices)[si].slice_id.resize(ids_size);
      for (size_t i = 0; i < grads.size(); i++) {
        const Slices& cur_slices = slices[i];
        const Tensor& cur_grad = grads[i];
        CASES(input_grad.Type(), do {
              std::vector<size_t> position;
              size_t filter_count;
              id_position->Get((int64_t*)&cur_slices.slice_id[0], cur_slices.slice_id.size(), true, 0.0, &position, &no_use, &filter_count);
              MultiThreadDoTBB(cur_slices.slice_id.size(), [&](const Range& r) {
                    for (size_t j = r.begin; j < r.end; j++) {
                      T* o = output_grad.Raw<T>(position[j]);
                      T* i = cur_grad.Raw<T>(j);
                      for (size_t k = 0; k < cur_slices.slice_size; k++) {
                        *o++ += *i++;
                      }
                      (*output_slices)[si].slice_id[position[j]] = cur_slices.slice_id[j];
                    }
                    return Status::Ok();
                  });
            } while(0));
      }

      CASES(input_grad.Type(), MultiThreadDoTBB(ids_size, [&](const Range& r) {
            for (size_t i = r.begin; i < r.end; i++) {
              T* out = output_grad.Raw<T>(i);
              for (size_t j = 0; j < input_slices.slice_size; j++) {
                *out++ /= worker_count;
              }
            }
            return Status::Ok();}));
      (*output_grads)[si] = std::move(output_grad);
      delete id_position;
    }
    return Status::Ok();
  }
private:
  static std::mutex lock_;
  static std::map<std::string, std::vector<Slices>> slices_map;
  static std::map<std::string, std::vector<Tensor>> grads_map;
  //map<token_name, pair<token, count>>
  static std::map<std::string, std::pair<int64_t, int> > current_tokens;
};

std::map<std::string, std::vector<Slices>> AggregateSlice::slices_map;
std::map<std::string, std::vector<Tensor>> AggregateSlice::grads_map;
std::map<std::string, std::pair<int64_t, int> > AggregateSlice::current_tokens;
std::mutex AggregateSlice::lock_;

SIMPLE_UDF_REGISTER(AggregateSlice, AggregateSlice);

}
}
}
