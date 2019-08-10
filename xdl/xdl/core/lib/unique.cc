/*
 * Copyright 1999-2018 Alibaba Group.
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

#include "xdl/core/lib/unique.h"

namespace xdl {
namespace functor {

template <typename I> struct UniqueID {
  std::vector<I> samples;
  std::vector<I> idxs;  
};

template <typename T, typename I>
void UniqueFunctor<CpuDevice, T, I>::operator()(CpuDevice* d,
                                                const Tensor& in,
                                                const Tensor& segment,
                                                Tensor* out,
                                                Tensor* out_index,
                                                Tensor* sample_index,
                                                Tensor* sample_segment) {
  size_t id_num = in.Shape()[0];
  size_t id_dim = in.Shape().Size() == 1 ? 1 : in.Shape()[1];
  size_t total_num = in.Shape().NumElements();
  I current_sample = 0;
  T* pin = in.Raw<T>();
  *out_index = Tensor(d, TensorShape({id_num}), DataTypeToEnum<I>::v());
  I* pindex = out_index->Raw<I>();
  I* psegment = segment.Raw<I>();
  *sample_index = Tensor(d, TensorShape({id_num}), DataTypeToEnum<I>::v());
  I* psample_index = sample_index->Raw<I>();

  if (id_dim == 1) {
    std::unordered_map<T, UniqueID<I> > uniq(id_num);
    for (I i = 0; i < id_num; ++i) {
      while (psegment[current_sample] == i) {
        ++current_sample;
      }
      auto iter = uniq.insert(std::make_pair(pin[i], UniqueID<I>{.samples={current_sample}, .idxs={i}}));
      if (!iter.second) {
        iter.first->second.samples.push_back(current_sample);
        iter.first->second.idxs.push_back(i);
      }
    }
    TensorShape out_shape({uniq.size()});
    *out = Tensor(d, out_shape, DataTypeToEnum<T>::v());
    *sample_segment = Tensor(d, out_shape, DataTypeToEnum<I>::v());
    I accum = 0;  
    T* pout = out->Raw<T>();
    I* psample_segment = sample_segment->Raw<I>();
    I j = 0;
    for (const auto& it : uniq) {
      pout[j] = it.first;
      for (size_t i = 0; i < it.second.samples.size(); ++i, ++accum) {
        psample_index[accum] = it.second.samples[i];
      }
      for (size_t i = 0; i < it.second.idxs.size(); ++i) {
        pindex[it.second.idxs[i]] = j;
      }
      psample_segment[j] = accum;
      j++;
    }
  } else {
    auto hash_fn = [pin](const size_t& index) {
      size_t x = std::hash<T>()(pin[index * 2]);
      size_t y = std::hash<T>()(pin[index * 2 + 1]);
      x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
      y = ((y & 0xFFFFFFFF00000000L) >> 32) + ((y & 0x00000000FFFFFFFFL) << 32);
      return x ^ y;
    };
    auto key_equal_fn = [pin](const size_t& lhs, const size_t& rhs) {
      return pin[lhs * 2] == pin[rhs * 2] &&
             pin[lhs * 2 + 1] == pin[rhs * 2 + 1];
    };

    std::unordered_map<size_t, UniqueID<I>, decltype(hash_fn), decltype(key_equal_fn)>
        uniq(id_num, hash_fn, key_equal_fn);

    for (I i = 0; i < id_num; ++i) {
      while (psegment[current_sample] == i) {
        ++current_sample;
      }
      auto iter = uniq.insert(std::make_pair(i, UniqueID<I>{.samples={current_sample}, .idxs={i}}));
      if (!iter.second) {
        iter.first->second.samples.push_back(current_sample);
        iter.first->second.idxs.push_back(i);
      }
    }
    TensorShape out_shape({uniq.size(), id_dim});
    *out = Tensor(d, out_shape, DataTypeToEnum<T>::v());
    TensorShape sample_segment_shape({uniq.size()});    
    *sample_segment = Tensor(d, sample_segment_shape, DataTypeToEnum<I>::v());
    I accum = 0;  
    T* pout = out->Raw<T>();
    I* psample_segment = sample_segment->Raw<I>();
    I j = 0;
    for (const auto& it : uniq) {
      pout[2*j] = pin[2*it.first];
      pout[2*j+1] = pin[2*it.first+1];
      for (size_t i = 0; i < it.second.samples.size(); ++i, ++accum) {
        psample_index[accum] = it.second.samples[i];
      }
      for (size_t i = 0; i < it.second.idxs.size(); ++i) {
        pindex[it.second.idxs[i]] = j;
      }
      psample_segment[j] = accum;
      j++;
    }
  }
}

template struct UniqueFunctor<CpuDevice, int64_t, int64_t>;
template struct UniqueFunctor<CpuDevice, int32_t, int32_t>;
template struct UniqueFunctor<CpuDevice, int64_t, int32_t>;
template struct UniqueFunctor<CpuDevice, int32_t, int64_t>;

}  // namespace functor

}  // namespace xdl
