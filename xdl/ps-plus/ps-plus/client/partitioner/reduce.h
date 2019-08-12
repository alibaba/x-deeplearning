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

#ifndef PS_PLUS_CLIENT_PARTITIONER_REDUCE_H_
#define PS_PLUS_CLIENT_PARTITIONER_REDUCE_H_

#include "ps-plus/client/partitioner.h"

namespace ps {
namespace client {
namespace partitioner {
namespace reduce_impl {

template <typename T, typename Functor>
class Reduce : public Partitioner {
 public:
  virtual Status Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) override {
    WrapperData<T>* raw_src = dynamic_cast<WrapperData<T>*>(src);
    if (raw_src == nullptr) {
      return Status::ArgumentError("Logic Partitioner Combine src should be bool");
    }

    if (output->get() == nullptr) {
      output->reset(new WrapperData<T>(Functor::default_value));
    } 

    WrapperData<T>* dst = dynamic_cast<WrapperData<T>*>(output->get());
    Functor::Run(raw_src->Internal(), &dst->Internal());

    return Status::Ok();
  }
};

struct AddFunctor {
  static constexpr int default_value = 0;
  template<typename T>
  static void Run(T x, T* y) {
    *y += x;
  }
};

struct MulFunctor {
  static constexpr int default_value = 1;
  template<typename T>
  static void Run(T x, T* y) {
    *y *= x;
  }
};

struct AndFunctor {
  static constexpr bool default_value = true;
  template<typename T>
  static void Run(T x, T* y) {
    *y &= x;
  }
};

struct OrFunctor {
  static constexpr bool default_value = false;
  template<typename T>
  static void Run(T x, T* y) {
    *y |= x;
  }
};

template<typename T>
class ReduceSum : public Reduce<T, AddFunctor> {};

template<typename T>
class ReduceMul : public Reduce<T, MulFunctor> {};

class ReduceAnd : public Reduce<bool, AndFunctor> {};

class ReduceOr : public Reduce<bool, OrFunctor> {};

}

using reduce_impl::ReduceSum;
using reduce_impl::ReduceMul;
using reduce_impl::ReduceAnd;
using reduce_impl::ReduceOr;

}
}
}

#endif

