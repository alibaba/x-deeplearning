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
#include "ps-plus/common/initializer/constant_initializer.h"

namespace ps {
namespace server {
namespace udf {

using std::vector;

class ScalarIntegerLogger : public SimpleUdf<vector<Slices>, vector<std::string>, vector<int64_t> > {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const vector<Slices>& sslices,
      const vector<std::string>& slot_names,
      const vector<int64_t>& pvals) const {
    if (sslices.size() != slot_names.size() || sslices.size() != pvals.size()) {
      return Status::ArgumentError("ScalarIntegerLogger: slices and other size not match");
    }
    for (size_t si = 0; si < sslices.size(); si++) {
      const Slices& slices = sslices[si];
      Tensor* data_tensor = slices.variable->GetData();
      std::string slot_name = slot_names[si];
      Tensor* t = slices.variable->GetVariableLikeSlot(slot_name, DataType::kInt64, TensorShape(), []{ return new initializer::ConstantInitializer(0); });
      int64_t val = pvals[si];
      CASES(data_tensor->Type(), MultiThreadDo(slices.slice_id.size(), [&](const Range& r) {
                for (size_t i = r.begin; i < r.end; i++) {
                  int64_t slice = slices.slice_id[i];
                  if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
                    continue;
                  }
                  int64_t* data = t->Raw<int64_t>(slice);
                  *data = val;
                }
                return Status::Ok();
              }));
    }
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(ScalarIntegerLogger, ScalarIntegerLogger);

}
}
}

