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
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/string_utils.h"

namespace ps {
namespace server {
namespace udf {

class HashVariableInitializer : public SimpleUdf<DataType, TensorShape, std::string, std::unique_ptr<Initializer>> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const DataType& dt,
      const TensorShape& shape,
      const std::string& extra_info,
      const std::unique_ptr<Initializer>& initializer) const {
    if (shape.IsScalar()) {
      return Status::ArgumentError("Hash Shape Should not be Scalar");
    }
    std::unordered_map<std::string, std::string> kvs = StringUtils::ParseMap(extra_info);
    bool hash64 = false;
    int32_t bloom_filter_threthold = 0;
    for (const auto iter : kvs) {
      if (iter.first == "hash64" && iter.second == "true") {
        hash64 = true;
      } else if (iter.first == "bloom_filter") {
        if (!StringUtils::strToInt32(iter.second.c_str(), bloom_filter_threthold)) {
          return Status::ArgumentError("HashVariableInitializer: bloom_filter not int "  + iter.first + "=" + iter.second);
        }
        if (bloom_filter_threthold >= 65500) {
          return Status::ArgumentError("HashVariableInitializer: bloom_filter_threthold too large, only support < 65500, "  + iter.first + "=" + iter.second);
        }
        GlobalBloomFilter::SetThrethold(bloom_filter_threthold);
      }
    }
    if (bloom_filter_threthold != 0) {
      LOG(INFO) << ctx->GetVariableName() << ", bloom_filter_threthold " << bloom_filter_threthold;
    }
    std::string var_name = ctx->GetVariableName();
    Variable* var;
    ps::Status status = GetStorageManager(ctx)->Get(var_name, &var);
    if (!status.IsOk()) {
      return ctx->GetStorageManager()->Set(var_name, [&]{
            HashMap* hashmap = nullptr;
            if (hash64) {
              hashmap = new HashMapImpl<int64_t>(shape[0]);
            } else {
              hashmap = new HashMapImpl<Hash128Key>(shape[0]);
            }
            hashmap->SetBloomFilterThrethold(bloom_filter_threthold);
            Variable* var = new Variable(new Tensor(dt, shape, initializer->Clone(), Tensor::TType::kSegment, true), new WrapperData<std::unique_ptr<HashMap> >(hashmap), var_name);
            var->SetRealInited(true);
            return var;
          });
    } else {
      std::unique_ptr<HashMap>& hashmap = dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(var->GetSlicer())->Internal();
      if (hashmap.get() == nullptr) {
        return Status::ArgumentError("hashmap is empty for " + var_name);
      }
      var->GetData()->SetInititalizer(initializer->Clone());
      var->GetData()->InitChunkFrom(hashmap->GetSize());
      hashmap->SetBloomFilterThrethold(bloom_filter_threthold);
      var->SetRealInited(true);
      return Status::Ok();
    }
  }
};

SIMPLE_UDF_REGISTER(HashVariableInitializer, HashVariableInitializer);

}
}
}

