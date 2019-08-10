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
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/thread_pool.h"

#include <cstring>

namespace ps {
namespace server {
namespace udf {

class IdentityIndexVariableInitializer : public SimpleUdf<DataType, TensorShape, size_t, Tensor> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const DataType& type,
      const TensorShape& shape,
      const size_t& offset,
      const Tensor& tensor) const {
    if (type != tensor.Type() || shape != tensor.Shape()) {
      return Status::ArgumentError("IdentityIndexVariableInitializer: type or shape mismatch for variable and tensor");
    }
    std::string var_name = ctx->GetVariableName();
    Variable* var;
    ps::Status status = GetStorageManager(ctx)->Get(var_name, &var);
    if (!status.IsOk()) {
      return ctx->GetStorageManager()->Set(var_name, [&]{
            Tensor* ret = new Tensor(type, shape, new initializer::NoneInitializer);
            QuickMemcpy(ret->Raw<char>(), tensor.Raw<char>(), SizeOfType(type) * shape.NumElements());
            Variable* var = new Variable(ret, new WrapperData<size_t>(offset), var_name);
            var->SetRealInited(true);
            return var;});
    } else {
      var->SetRealInited(true);
      return Status::Ok();
    }
  }
};

SIMPLE_UDF_REGISTER(IdentityIndexVariableInitializer, IdentityIndexVariableInitializer);

}
}
}

