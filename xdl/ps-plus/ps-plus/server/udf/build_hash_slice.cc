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
#include "ps-plus/common/string_utils.h"
#include "ps-plus/common/logging.h"
#include <iostream>

namespace ps {
namespace server {
namespace udf {

class BuildHashSlice : public SimpleUdf<std::vector<Tensor>, std::vector<std::string>, std::vector<float>, bool, bool, std::vector<Slices>*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const std::vector<Tensor>& ids, const std::vector<std::string>& tensor_names, const std::vector<float>& save_ratios, const bool& writable, const bool& insert, std::vector<Slices>* result) const {
    static size_t step = 0;
    size_t current_step = step++;
    if (ids.size() != tensor_names.size()) {
      return Status::ArgumentError("BuildHashSlice: ids and tensor_names can't match");
    }
    StorageManager* manager = ctx->GetStorageManager();
    result->resize(ids.size());
    size_t total_id = 0;
    for (size_t si = 0; si < ids.size(); si++) {
      const Tensor& id = ids[si];
      Variable* variable;
      PS_CHECK_STATUS(manager->Get(tensor_names[si], &variable));
      if (id.Type() != DataType::kInt64) {
        return Status::ArgumentError("BuildHashSlice: dtype should be int64 for " + tensor_names[si]);
      }
      std::unique_ptr<HashMap>& hashmap = (dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(variable->GetSlicer()))->Internal();
      if (hashmap == nullptr) {
        return Status::ArgumentError("BuildHashSlice: Variable Should be a Hash Variable for " + tensor_names[si]);
      }
      QRWLocker locker(variable->VariableLock(), QRWLocker::kSimpleRead);
      Slices& element = (*result)[si];
      tbb::concurrent_vector<size_t> reused_ids;
      size_t total_filtered_count = 0;
      int64_t max_id = hashmap->Get(id.Raw<int64_t>(), id.Shape()[0], !insert, save_ratios[si], &element.slice_id, &reused_ids, &total_filtered_count);
      if (!insert && (current_step % 200 == 0) && total_filtered_count != 0) {
        LOG(INFO) << "Step " << current_step/2 << ", variable[" << tensor_names[si] << "], filtered keys [" << total_filtered_count << "], hashmap size [" << hashmap->GetSize() << "]";
      }
      if (max_id > 0) {
        PS_CHECK_STATUS(variable->ReShapeId(max_id));
      }
      if (reused_ids.size() != 0) {
        std::vector<size_t> raw_reused_ids;
        for (auto iter : reused_ids) {
          raw_reused_ids.push_back(iter);
        }
        variable->ClearIds(raw_reused_ids);
      }
      ps::TensorShape shape = variable->GetData()->Shape();
      element.slice_size = shape.NumElements() / shape[0];
      element.writable = writable;
      element.variable = variable;
      element.dim_part = 1;
      if (writable && ctx->GetStreamingModelArgs() != NULL && !ctx->GetStreamingModelArgs()->streaming_hash_model_addr.empty()) { 
          PS_CHECK_STATUS(StreamingModelUtils::WriteHash(tensor_names[si], id));
      }
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(BuildHashSlice, BuildHashSlice);

}
}
}

