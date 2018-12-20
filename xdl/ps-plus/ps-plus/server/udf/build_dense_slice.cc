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
#include "ps-plus/server/streaming_model_utils.h"

namespace ps {
namespace server {
namespace udf {

class BuildDenseSlice : public SimpleUdf<bool, Slices*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const bool& writable, Slices* result) const {
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("BuildDenseSlice: Variable should not be empty");
    }
    result->writable = writable;
    result->variable = variable;
    result->dim_part = -1;
    result->slice_size = variable->GetData()->Shape().NumElements();
    result->slice_id.push_back(0);

    if (writable && !ctx->GetStreamingModelArgs()->streaming_dense_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::WriteDense(ctx->GetVariableName()));
    }

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(BuildDenseSlice, BuildDenseSlice);

}
}
}

