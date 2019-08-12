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

struct Indice {
  unsigned int i;
  unsigned int j;
  Indice(size_t i, size_t j) : i(i), j(j) {}
};

template <typename T, typename I>
void HostAddSparse(const std::vector<Tensor>& in_grads,
                   const std::vector<Tensor>& in_ids,
                   Tensor* out_grads, Tensor* out_ids) {
  size_t count = 0;
  std::vector<std::vector<Indice>> indices_vec;
  bool is_hash64 = (in_ids[0].Shape().Size() == 1);
  if (is_hash64) {
    std::unordered_map<I, size_t> uniq_map;
    for (size_t i = 0; i < in_ids.size(); ++i) {
      size_t n = in_ids[i].Shape()[0];
      I* pid = in_ids[i].Raw<I>();
      for (size_t j = 0; j < n; ++j) {
        auto iter = uniq_map.insert(std::make_pair(pid[j], count));
        if (iter.second) {
          ++count;
          indices_vec.push_back(std::vector<Indice>{Indice(i, j)});
        } else {
          indices_vec[iter.first->second].push_back(Indice(i, j));
        }
      }
    }
  } else {
    auto hash_fn = [](const std::pair<I, I>& id) {
      size_t x = std::hash<I>()(id.first);
      size_t y = std::hash<I>()(id.second);
      x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
      y = ((y & 0xCCCCCCCCCCCCCCCCL) >> 2) + ((y & 0x3333333333333333L) << 2);
      return x ^ y;
    };
    auto equal_fn = [](const std::pair<I, I>& lhs,
                       const std::pair<I, I>& rhs) {
      return lhs.first == rhs.first && lhs.second == rhs.second;
    };
    std::unordered_map<std::pair<I, I>, size_t, decltype(hash_fn), decltype(equal_fn)> uniq_map(0, hash_fn, equal_fn);
    for (size_t i = 0; i < in_ids.size(); ++i) {
      size_t n = in_ids[i].Shape()[0];
      I* pid = in_ids[i].Raw<I>();
      for (size_t j = 0; j < n; ++j) {
        auto iter = uniq_map.insert(std::make_pair(std::make_pair(pid[j*2], pid[j*2+1]), count));
        if (iter.second) {
          ++count;
          indices_vec.push_back(std::vector<Indice>{Indice(i, j)});
        } else {
          indices_vec[iter.first->second].push_back(Indice(i, j));
        }
      }
    }
  }
  TensorShape id_shape = in_ids[0].Shape();
  id_shape.Set(0, count);
  *out_ids = Tensor(DeviceSingleton::CpuInstance(), id_shape, in_ids[0].Type());
  I* pid_out = out_ids->Raw<I>();

  size_t eb_dim = in_grads[0].Shape()[1];
  TensorShape uniq_shape({count, eb_dim});
  *out_grads = Tensor(DeviceSingleton::CpuInstance(), uniq_shape,
                      in_grads[0].Type());
  T* puniq = out_grads->Raw<T>();
  #pragma omp parallel for
  for (size_t c = 0; c < count; ++c) {
    const std::vector<Indice>& indices = indices_vec[c];
    size_t s = 0;
    const unsigned int i = indices[s].i, j = indices[s].j;
    if (is_hash64) {
      pid_out[c] = in_ids[i].Raw<I>()[j];
    } else {
      pid_out[c*2] = in_ids[i].Raw<I>()[j*2];
      pid_out[c*2+1] = in_ids[i].Raw<I>()[j*2+1];
    }
    T* p = puniq + c*eb_dim;
    T* pgrad = in_grads[i].Raw<T>() + j*eb_dim;
    memcpy(p, pgrad, eb_dim * sizeof(T));
    //for (size_t k = 0; k < eb_dim; ++k)  p[k] = pgrad[k];
    for (s = 1; s < indices.size(); ++s) {
      T* pgrad = in_grads[indices[s].i].Raw<T>() + indices[s].j*eb_dim;
      for (size_t k = 0; k < eb_dim; ++k) {
        p[k] += pgrad[k];
      }
    }
  }
}

}  // namespace xdl

#endif  // XDL_CORE_OPS_ADD_SPARSE_GRADIENT_H_
