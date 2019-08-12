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

namespace ps {
namespace server {
namespace udf {

class IndexVariableInitializer : public SimpleUdf<DataType, TensorShape, size_t, std::unique_ptr<Initializer>> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const DataType& type,
      const TensorShape& shape,
      const size_t& offset,
      const std::unique_ptr<Initializer>& initializer) const {
    std::string var_name = ctx->GetVariableName();
    Variable* var;
    ps::Status status = GetStorageManager(ctx)->Get(var_name, &var);
    if (!status.IsOk()) {
      return ctx->GetStorageManager()->Set(var_name, [&]{ Variable* var = new Variable(new Tensor(type, shape, initializer->Clone()), new WrapperData<size_t>(offset), var_name); var->SetRealInited(true); return var;});
    } else {
      var->SetRealInited(true);
      return Status::Ok();
    }
  }
};

SIMPLE_UDF_REGISTER(IndexVariableInitializer, IndexVariableInitializer);

}
}
}

