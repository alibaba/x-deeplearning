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

class BuildSparseSlice : public SimpleUdf<Tensor, bool, std::vector<Slices>*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const Tensor& ids, const bool& writable, std::vector<Slices>* result) const {
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("BuildSparseSlice: Variable should not be empty");
    }
    if (variable->GetData()->Shape().IsScalar()) {
      return Status::ArgumentError("BuildSparseSlice: Variable should not be Scalar");
    }
    if (ids.Shape().Size() != 1) {
      return Status::ArgumentError("BuildSparseSlice: Id should be a vector");
    }
    WrapperData<size_t>* offset = dynamic_cast<WrapperData<size_t>*>(variable->GetSlicer());
    if (offset == nullptr) {
      return Status::ArgumentError("BuildSparseSlice: Variable Should be a Indexed Variable");
    }
    int64_t min_id = offset->Internal();
    int64_t max_id = variable->GetData()->Shape()[0] + min_id;

    Slices slices;
    slices.writable = writable;
    slices.variable = variable;
    slices.dim_part = 1;
    slices.slice_size = variable->GetData()->Shape().NumElements() / variable->GetData()->Shape()[0];
    TensorShape shape = variable->GetData()->Shape();
    CASES(ids.Type(), do {
                            slices.slice_id.reserve(ids.Shape()[0]);
                            for (size_t i = 0; i < ids.Shape()[0]; i++) {
                              int64_t id = ids.Raw<T>()[i];
                              if (id < min_id || id >= max_id) {
                                return Status::ArgumentError("BuildSparseSlice: id Overflow");
                              }
                              slices.slice_id.push_back(id - min_id);
                            }
                          } while (0));
    result->push_back(slices);
    //TODO Write Sparse
    if (writable && ctx->GetStreamingModelArgs() != NULL  && !ctx->GetStreamingModelArgs()->streaming_sparse_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::WriteSparse(ctx->GetVariableName(), ids));
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(BuildSparseSlice, BuildSparseSlice);

}
}
}

