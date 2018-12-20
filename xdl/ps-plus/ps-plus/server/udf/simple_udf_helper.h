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

#ifndef PS_SERVER_UDF_SIMPLE_UDF_HELPER_H_
#define PS_SERVER_UDF_SIMPLE_UDF_HELPER_H_

#include "ps-plus/common/data.h"

namespace ps {
namespace server {
namespace udf {
namespace simple_udf_helper {

struct Input {};
struct Output {};

template<typename T>
struct Argument {
  using Type = Input;
  using PointerType = T*;
  using ArgType = const T&;
  static bool Check(Data* data) {
    return dynamic_cast<WrapperData<T>*>(data) != nullptr;
  }
  static ArgType ToArg(Data* data) {
    return dynamic_cast<WrapperData<T>*>(data)->Internal();
  }
  static Data* Build() {
    return nullptr;
  }
};

template<typename T>
struct Argument<T*> {
  using Type = Output;
  using PointerType = T*;
  using ArgType = T*;
  static bool Check(Data* data) {
    return true;
  }
  static ArgType ToArg(Data* data) {
    return &dynamic_cast<WrapperData<T>*>(data)->Internal();
  }
  static Data* Build() {
    return new WrapperData<T>;
  }
};

template<typename... T>
struct CounterImpl {};

template<typename... T>
struct CounterImpl<Input, Input, T...> {
  using Internal = CounterImpl<Input, T...>;
  static constexpr size_t kInputSize = Internal::kInputSize + 1;
  static constexpr size_t kOutputSize = Internal::kOutputSize;
};

template<typename... T>
struct CounterImpl<Input, Output, T...> {
  using Internal = CounterImpl<Output, T...>;
  static constexpr size_t kInputSize = Internal::kInputSize + 1;
  static constexpr size_t kOutputSize = Internal::kOutputSize;
};

template<typename... T>
struct CounterImpl<Output, Output, T...> {
  using Internal = CounterImpl<Output, T...>;
  static constexpr size_t kInputSize = Internal::kInputSize;
  static constexpr size_t kOutputSize = Internal::kOutputSize + 1;
};

template<>
struct CounterImpl<Input> {
  static constexpr size_t kInputSize = 1;
  static constexpr size_t kOutputSize = 0;
};

template<>
struct CounterImpl<Output> {
  static constexpr size_t kInputSize = 0;
  static constexpr size_t kOutputSize = 1;
};

template<typename... T>
struct Counter {
  using Impl = CounterImpl<typename Argument<T>::Type...>;
  static constexpr size_t kInputSize = Impl::kInputSize;
  static constexpr size_t kOutputSize = Impl::kOutputSize;
};

}
}
}
}

// Following Code is build by codegen
// template<typename Tudf, typename...>
// Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas);
// Following is the codegen python code:
//
// t0 = "typename T%d"
// t1 = "Argument<T%d>::Check(datas[%d])"
// t2 = "Argument<T%d>::ToArg(datas[%d])"
// t3 = """
// template<%s>
// Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
//   if (!(%s)) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
//     return udf->SimpleRun(%s);
//   };
// """
//
// for kk in range(32):
//   tx0 = ', '.join(["typename Tudf"] + [t0 % i for i in range(kk)])
//   tx1 = ' && '.join(["true"] + [t1 % (i, i) for i in range(kk)])
//   tx2 = ', '.join(["ctx"] + [t2 % (i, i) for i in range(kk)])
//   print t3 % (tx0, tx1, tx2)

namespace ps {
namespace server {
namespace udf {
namespace simple_udf_helper {

template<typename Tudf>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true)) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx);
};


template<typename Tudf, typename T0>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]));
};


template<typename Tudf, typename T0, typename T1>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]));
};


template<typename Tudf, typename T0, typename T1, typename T2>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]) && Argument<T26>::Check(datas[26]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]), Argument<T26>::ToArg(datas[26]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]) && Argument<T26>::Check(datas[26]) && Argument<T27>::Check(datas[27]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]), Argument<T26>::ToArg(datas[26]), Argument<T27>::ToArg(datas[27]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]) && Argument<T26>::Check(datas[26]) && Argument<T27>::Check(datas[27]) && Argument<T28>::Check(datas[28]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]), Argument<T26>::ToArg(datas[26]), Argument<T27>::ToArg(datas[27]), Argument<T28>::ToArg(datas[28]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]) && Argument<T26>::Check(datas[26]) && Argument<T27>::Check(datas[27]) && Argument<T28>::Check(datas[28]) && Argument<T29>::Check(datas[29]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]), Argument<T26>::ToArg(datas[26]), Argument<T27>::ToArg(datas[27]), Argument<T28>::ToArg(datas[28]), Argument<T29>::ToArg(datas[29]));
};


template<typename Tudf, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30>
Status SimpleRunHelper(const Tudf* udf, UdfContext* ctx, const std::vector<Data*>& datas) {
  if (!(true && Argument<T0>::Check(datas[0]) && Argument<T1>::Check(datas[1]) && Argument<T2>::Check(datas[2]) && Argument<T3>::Check(datas[3]) && Argument<T4>::Check(datas[4]) && Argument<T5>::Check(datas[5]) && Argument<T6>::Check(datas[6]) && Argument<T7>::Check(datas[7]) && Argument<T8>::Check(datas[8]) && Argument<T9>::Check(datas[9]) && Argument<T10>::Check(datas[10]) && Argument<T11>::Check(datas[11]) && Argument<T12>::Check(datas[12]) && Argument<T13>::Check(datas[13]) && Argument<T14>::Check(datas[14]) && Argument<T15>::Check(datas[15]) && Argument<T16>::Check(datas[16]) && Argument<T17>::Check(datas[17]) && Argument<T18>::Check(datas[18]) && Argument<T19>::Check(datas[19]) && Argument<T20>::Check(datas[20]) && Argument<T21>::Check(datas[21]) && Argument<T22>::Check(datas[22]) && Argument<T23>::Check(datas[23]) && Argument<T24>::Check(datas[24]) && Argument<T25>::Check(datas[25]) && Argument<T26>::Check(datas[26]) && Argument<T27>::Check(datas[27]) && Argument<T28>::Check(datas[28]) && Argument<T29>::Check(datas[29]) && Argument<T30>::Check(datas[30]))) { return Status::ArgumentError("SimpleUdf: Argument Data Error"); }
  return udf->SimpleRun(ctx, Argument<T0>::ToArg(datas[0]), Argument<T1>::ToArg(datas[1]), Argument<T2>::ToArg(datas[2]), Argument<T3>::ToArg(datas[3]), Argument<T4>::ToArg(datas[4]), Argument<T5>::ToArg(datas[5]), Argument<T6>::ToArg(datas[6]), Argument<T7>::ToArg(datas[7]), Argument<T8>::ToArg(datas[8]), Argument<T9>::ToArg(datas[9]), Argument<T10>::ToArg(datas[10]), Argument<T11>::ToArg(datas[11]), Argument<T12>::ToArg(datas[12]), Argument<T13>::ToArg(datas[13]), Argument<T14>::ToArg(datas[14]), Argument<T15>::ToArg(datas[15]), Argument<T16>::ToArg(datas[16]), Argument<T17>::ToArg(datas[17]), Argument<T18>::ToArg(datas[18]), Argument<T19>::ToArg(datas[19]), Argument<T20>::ToArg(datas[20]), Argument<T21>::ToArg(datas[21]), Argument<T22>::ToArg(datas[22]), Argument<T23>::ToArg(datas[23]), Argument<T24>::ToArg(datas[24]), Argument<T25>::ToArg(datas[25]), Argument<T26>::ToArg(datas[26]), Argument<T27>::ToArg(datas[27]), Argument<T28>::ToArg(datas[28]), Argument<T29>::ToArg(datas[29]), Argument<T30>::ToArg(datas[30]));
};

}
}
}
}

#endif

