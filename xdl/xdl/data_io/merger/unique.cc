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

#include "xdl/data_io/merger/unique.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace functor {

template <typename T, typename I>
void UniqueFunctor<CpuDevice, T, I>::operator()(CpuDevice* d,
                                                const Tensor &in,
                                                Tensor *out,
                                                Tensor *out_index) {
  auto shape = in.Shape();
  XDL_CHECK(shape.Size() == 1 || shape.Size() == 2);
  size_t id_num = shape[0];
  size_t id_dim = shape.Size() == 1 ? 1 : shape[1];
  size_t total_num = shape.NumElements();
  XDL_CHECK(in.Type() == DataTypeToEnum<T>::v());
  T* pin = in.Raw<T>();
  *out_index = Tensor(d, TensorShape({id_num}), DataTypeToEnum<I>::v());
  I* pindex = out_index->Raw<I>();
  if (id_dim == 1) {
    std::unordered_map<T, I> uniq(id_num);
    I j = 0;
    for (size_t i = 0; i < id_num; ++i) {
      auto iter = uniq.insert(std::make_pair(pin[i], j));
      pindex[i] = iter.first->second;
      if (iter.second) {
        ++j;
      }
    }
    TensorShape out_shape({uniq.size()});
    *out = Tensor(d, out_shape, DataTypeToEnum<T>::v());

    T* buf = out->Raw<T>();
    for (const auto& it : uniq) {
      buf[it.second] = it.first;
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

    std::unordered_map<size_t, I, decltype(hash_fn), decltype(key_equal_fn)>
        uniq(id_num, hash_fn, key_equal_fn);
    I j = 0;
    for(size_t i = 0; i < id_num; ++i) {
      auto iter = uniq.insert(std::make_pair(i, j));
      pindex[i] = iter.first->second;
      if (iter.second) {
        ++j;
      }
    }
    TensorShape out_shape({uniq.size(), id_dim});
    *out = Tensor(d, out_shape, DataTypeToEnum<T>::v());

    T* buf = out->Raw<T>();
    for (const auto& it : uniq) {
      buf[it.second * 2] = pin[it.first * 2];
      buf[it.second * 2 + 1] = pin[it.first * 2 + 1];
    }
  }
}

template struct UniqueFunctor<CpuDevice, int64_t, int64_t>;
template struct UniqueFunctor<CpuDevice, int32_t, int32_t>;
template struct UniqueFunctor<CpuDevice, int64_t, int32_t>;
template struct UniqueFunctor<CpuDevice, int32_t, int64_t>;

}  // namespace functor

/*
template <typename T, typename I>
Status UniqueCpuOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor input, output, out_index;
  XDL_CHECK_STATUS(ctx->GetInput(0, &input));
  XDL_CHECK_COND(2 >= input.Shape().Size(),
                 Status::ArgumentError("input dim cann't be greater than 2"));
  TensorShape index_shape({input.Shape()[0]});
  XDL_CHECK_STATUS(ctx->AllocateOutput(1, index_shape, &out_index));

  CpuDevice* device = dynamic_cast<CpuDevice*>(ctx->GetDevice());
  auto fn = functor::UniqueFunctor<CpuDevice, T, I>();
  fn(device, input, &output, out_index);

  ctx->SetOutput(0, output);
  return Status::Ok();
}
*/

}  // namespace xdl
