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
#include "ps-plus/common/hashmap.h"
#include "ps-plus/server/streaming_model_utils.h"
#include <iostream>

namespace ps {
namespace server {
namespace udf {

class BuildHashSlice : public SimpleUdf<Tensor, bool, double, Slices*> {
 public:
 virtual Status SimpleRun(UdfContext* ctx, const Tensor& ids, const bool& writable, const double& add_probability, Slices* result) const {
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("BuildHashSlice: Variable should not be empty");
    }
    if (variable->GetData()->Shape().IsScalar()) {
      return Status::ArgumentError("BuildHashSlice: Variable should not be Scalar");
    }
    if (ids.Shape().Size() != 2 || ids.Shape()[1] != 2 || ids.Type() != DataType::kInt64) {
      return Status::ArgumentError("BuildHashSlice: Id should be [?:2] and dtype should be int64");
    }
    WrapperData<HashMap>* hashmap = dynamic_cast<WrapperData<HashMap>*>(variable->GetSlicer());
    if (hashmap == nullptr) {
      return Status::ArgumentError("BuildHashSlice: Variable Should be a Hash Variable");
    }

    result->writable = writable;
    result->variable = variable;
    result->dim_part = 1;
    result->slice_size = variable->GetData()->Shape().NumElements() / variable->GetData()->Shape()[0];

    std::vector<int64_t> raw_ids, reused_ids;
    hashmap->Internal().GetWithAddProbability(ids.Raw<int64_t>(), ids.Shape()[0], 2, add_probability, &raw_ids, &reused_ids);
    std::vector<size_t> raw_reused_ids;
    for (int64_t id : reused_ids) {
      raw_reused_ids.push_back(id);
    }
    variable->ClearIds(raw_reused_ids);
    size_t max = 0;
    result->slice_id.reserve(raw_ids.size());
    for (int64_t id : raw_ids) {
      result->slice_id.push_back(id);
      if (id > 0) {
        max = std::max(max, (size_t)id);
      }
    }
    if (max >= variable->GetData()->Shape()[0]) {
      ctx->GetLocker()->ChangeType(QRWLocker::kWrite);
      if (max >= variable->GetData()->Shape()[0]) {
        TensorShape shape = variable->GetData()->Shape();
        size_t x = shape[0];
        while (x <= max) {
          x *= 2;
        }
        PS_CHECK_STATUS(variable->ReShapeId(x));
      }
      ctx->GetLocker()->ChangeType(QRWLocker::kSimpleRead);
    }
    if (writable && !ctx->GetStreamingModelArgs()->streaming_hash_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::WriteHash(ctx->GetVariableName(), ids));
    }

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(BuildHashSlice, BuildHashSlice);

}
}
}

