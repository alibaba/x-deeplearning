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

namespace ps {
namespace server {
namespace udf {

class TransSlice : public SimpleUdf<std::vector<Slices>, TensorSlices*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const std::vector<Slices>& slices, TensorSlices* result) const {
    if (slices.size() != 1) {
      return Status::ArgumentError("TransSlice: slices size must be 1");
    }
    result->slice_size = slices[0].slice_size;
    result->slice_id = slices[0].slice_id;
    result->dim_part = slices[0].dim_part;
    result->tensor = *(slices[0].variable->GetData());
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(TransSlice, TransSlice);

}
}
}

