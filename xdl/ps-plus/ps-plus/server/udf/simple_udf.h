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

#ifndef PS_SERVER_UDF_SIMPLE_UDF_H_
#define PS_SERVER_UDF_SIMPLE_UDF_H_

#include "ps-plus/server/udf.h"
#include "ps-plus/server/udf/simple_udf_helper.h"

namespace ps {
namespace server {
namespace udf {

// You can use this to build a udf simply.
// If You have a udf, input for WrapperData<input1>, WrapperData<input2>
// And output for WrapperData<output1>, WrapperData<output2>
// You can just inheritance SimpleUdf<input1, input2, output1*, output2*>,
// and implement the virtual function SimpleRun(const input1&, const input2&, output1*, output2*).
// Then you can get your udf.
//
// Example:
// class AddOperator : public SimpleUdf<int, int, int*> {
//   virtual Status SimpleRun(UdfContext* ctx, const int& x, const int& y, int* z) const override {
//     *z = x + y;
//     return Status::Ok();
//   };
// };

template <typename... T>
class SimpleUdf : public Udf {
 public:
  virtual Status SimpleRun(UdfContext* ctx, typename simple_udf_helper::Argument<T>::ArgType...) const = 0;

  virtual Status Run(UdfContext* ctx) const {
    if (InputSize() != kInputSize || OutputSize() != kOutputSize) {
      return Status::ArgumentError("Simple Udf InputSize or OutputSize Error!");
    }
    std::vector<Data*> inputs;
    PS_CHECK_STATUS(GetInputs(ctx, &inputs));
    std::vector<Data*> args({simple_udf_helper::Argument<T>::Build()...});
    for (size_t i = 0; i < kInputSize; i++) {
      args[i] = inputs[i];
    }
    Status ret = simple_udf_helper::SimpleRunHelper<SimpleUdf, T...>(this, ctx, args);
    if (ret.IsOk()) {
      for (size_t i = 0; i < kOutputSize; i++) {
        Udf::SetOutput(ctx, i, args[kInputSize + i]);
      }
    } else {
      for (size_t i = 0; i < kOutputSize; i++) {
        delete args[kInputSize + i];
      }
    }
    return ret;
  }

  static constexpr size_t kInputSize = simple_udf_helper::Counter<T...>::kInputSize;
  static constexpr size_t kOutputSize = simple_udf_helper::Counter<T...>::kOutputSize;
};

}
}
}

#define SIMPLE_UDF_REGISTER(TYPE, NAME) \
  UDF_REGISTER(TYPE, NAME, TYPE::kInputSize, TYPE::kOutputSize)

#define CHECK_COUNTER(COUNTER, OK) do { if (--COUNTER == 0) {OK.set_value(true);} return;} while(0);

#endif

