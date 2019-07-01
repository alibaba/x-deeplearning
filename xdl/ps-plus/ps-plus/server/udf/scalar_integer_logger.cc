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

class ScalarIntegerLogger : public SimpleUdf<Slices, std::string, int64_t> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const std::string& slot_name,
      const int64_t& pval) const {
    Tensor* t = slices.variable->GetVariableLikeSlot(slot_name, DataType::kInt64, TensorShape(), []{ return new initializer::ConstantInitializer(0); });
    int64_t* data = t->Raw<int64_t>();
    int64_t val = pval;
    for (size_t slice : slices.slice_id) {
      if (slice != (size_t)HashMap::NOT_ADD_ID) {
        data[slice] = val;
      }
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(ScalarIntegerLogger, ScalarIntegerLogger);

}
}
}

