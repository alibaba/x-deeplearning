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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_OPS_ADD_SPARSE_GRADIENT_H_
#define XDL_CORE_OPS_ADD_SPARSE_GRADIENT_H_

#include <omp.h>
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/backend/device_singleton.h"

namespace xdl {

template <typename T, typename I>
void HostAddSparse(const std::vector<Tensor>& in_grads,
                   const std::vector<Tensor>& in_ids,
                   Tensor* out_grads, Tensor* out_ids) {
  std::vector<std::vector<size_t>> indices;
  size_t count = 0;

  if (in_ids[0].Shape().Size() == 1) {
    // make unique id
    std::unordered_map<I, size_t> uniq_map;
    for (size_t i = 0; i < in_ids.size(); ++i) {
      size_t n = in_ids[i].Shape()[0];
      I* pid = in_ids[i].Raw<I>();
      std::vector<size_t> index;
      index.reserve(n);
      for (size_t k = 0; k < n; ++k) {
        auto iter = uniq_map.insert(std::make_pair(pid[k], count));
        index.push_back(iter.first->second);
        if (iter.second) ++count;
      }
      indices.push_back(index);
    }
    TensorShape id_shape = in_ids[0].Shape();
    id_shape.Set(0, count);
    *out_ids = Tensor(DeviceSingleton::CpuInstance(), id_shape,
                      in_ids[0].Type());
    I* pid = out_ids->Raw<I>();
    for (auto&& iter : uniq_map) {
      pid[iter.second] = iter.first;
    }
  } else {
    // make unique id
    auto hash_fn = [](const std::pair<I, I>& id) {
      size_t x = std::hash<I>()(id.first);
      size_t y = std::hash<I>()(id.second);
      x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
      y = ((y & 0xFFFFFFFF00000000L) >> 32) + ((y & 0x00000000FFFFFFFFL) << 32);
      return x ^ y;
    };
    auto equal_fn = [](const std::pair<I, I>& lhs,
                       const std::pair<I, I>& rhs) {
      return lhs.first == rhs.first && lhs.second == rhs.second;
    };
    std::unordered_map<std::pair<I, I>, size_t, decltype(hash_fn),
        decltype(equal_fn)> uniq_map(0, hash_fn, equal_fn);
    for (size_t i = 0; i < in_ids.size(); ++i) {
      size_t n = in_ids[i].Shape()[0];
      I* pid = in_ids[i].Raw<I>();
      std::vector<size_t> index;
      index.reserve(n);
      for (size_t k = 0; k < n; ++k) {
        auto key = std::make_pair(pid[2 * k], pid[2 * k + 1]);
        auto iter = uniq_map.insert(std::make_pair(key, count));
        index.push_back(iter.first->second);
        if (iter.second) ++count;
      }
      indices.push_back(index);
    }
    TensorShape id_shape = in_ids[0].Shape();
    id_shape.Set(0, count);
    *out_ids = Tensor(DeviceSingleton::CpuInstance(), id_shape,
                      in_ids[0].Type());
    I* pid = out_ids->Raw<I>();
    for (auto&& iter : uniq_map) {
      pid[2*iter.second] = iter.first.first;
      pid[2*iter.second+1] = iter.first.second;
    }
  }
  // add to unique grad
  size_t eb_dim = in_grads[0].Shape()[1];
  TensorShape uniq_shape({count, eb_dim});
  *out_grads = Tensor(DeviceSingleton::CpuInstance(), uniq_shape,
                      in_grads[0].Type());
  T* puniq = out_grads->Raw<T>();
  std::memset(puniq, 0, sizeof(T) * uniq_shape.NumElements());
  for (size_t i = 0; i < in_ids.size(); ++i) {
    size_t n = in_ids[i].Shape()[0];
    T* pgrad = in_grads[i].Raw<T>();
    I* pid = in_ids[i].Raw<I>();
//    #pragma omp parallel for
    for (size_t j = 0; j < n; ++j) {
      size_t idx = indices[i][j];
      for (size_t k = 0; k < eb_dim; ++k) {
        common::cpu_atomic_add<T>(pgrad[j*eb_dim+k], puniq+idx*eb_dim+k);
      }
    }
  }
}

}  // namespace xdl

#endif  // XDL_CORE_OPS_ADD_SPARSE_GRADIENT_H_
