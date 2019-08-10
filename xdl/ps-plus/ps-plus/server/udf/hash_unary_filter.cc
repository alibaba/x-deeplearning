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
#include "ps-plus/common/logging.h"

namespace ps {
namespace server {
namespace udf {

#define OPS_SINGLE_ARGS(...) __VA_ARGS__

#define OPS_PROCESS(STMT) \
  do {                                       \
    OP_PROCESS(i, >=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, >=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, <=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, <=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, >, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(d, >, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(i, <, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(d, <, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(i, ==, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, ==, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, !=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, !=, OPS_SINGLE_ARGS(STMT))  \
    return Status::ArgumentError("HashUnaryFilter cond Error"); \
  } while (0)

#define OP_PROCESS(ARG, OP, STMT)                    \
  {                                                 \
    static const std::string ARGOP = #ARG #OP;      \
    if (ARGOP.size() <= cond.size() && cond.substr(0, ARGOP.size()) == ARGOP) { \
      auto TESTFN = [=](decltype(ARG) x) {          \
        return ARG OP x;                            \
      };                                            \
      std::string SLOT = cond.substr(ARGOP.size()); \
      STMT                                          \
      break;                                        \
    }                                               \
  }



class HashUnaryFilter : public SimpleUdf<std::string, double, int64_t> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const std::string& cond, const double& pd, const int64_t& pi) const {
    #if 0
    double d = pd;
    int64_t i = pi;
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("HashUnaryFilter: Variable should not be empty");
    }
    if (variable->GetData()->Shape().IsScalar()) {
      return Status::ArgumentError("HashUnaryFilter: Variable should not be Scalar");
    }
    WrapperData<HashMap>* hashmap = dynamic_cast<WrapperData<HashMap>*>(variable->GetSlicer());
    if (hashmap == nullptr) {
      return Status::ArgumentError("HashUnaryFilter: Variable Should be a Hash Variable");
    }
    HashMap::HashMapStruct map;
    if (hashmap->Internal().GetHashKeys(&map) != 0) {
      return Status::Unknown("HashUnaryFilter Get Gasg Keys Error");
    }

    std::vector<int64_t> keys;

    OPS_PROCESS(
      Tensor* tensor;
      if (SLOT == "_") {
        tensor = variable->GetData();
      } else {
        PS_CHECK_STATUS(variable->GetExistSlot(SLOT, &tensor));
      }
      auto s = tensor->Shape().Dims();
      if (s.size() != 1) {
        return Status::ArgumentError("HashUnaryFilter: Slot Shape Error! Must be 1 dim");
      }
      CASES(tensor->Type(), {
        T* data = tensor->Raw<T>();
        for (auto& item : map.items) {
          if (TESTFN(data[item.id])) {
            keys.push_back(item.x);
            keys.push_back(item.y);
          }
        }
      });
    );

    hashmap->Internal().Del(&(keys[0]), keys.size() / 2, 2);

    if (!ctx->GetStreamingModelArgs()->streaming_hash_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::DelHash(ctx->GetVariableName(), keys));
    }

    LOG_INFO("Hash Filter for %s, origin=%lld, clear=%lld", ctx->GetVariableName().c_str(), map.items.size(), keys.size() / 2);
    #endif 
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(HashUnaryFilter, HashUnaryFilter);

}
}
}

