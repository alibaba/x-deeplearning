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
#include "ps-plus/server/udf/python_runner.h"
#include "ps-plus/common/initializer/constant_initializer.h"

namespace ps {
namespace server {
namespace udf {

namespace {

struct Argument {
  enum Type {
    kArr,
    kSlice
  };
  Type type;
  Tensor t;
  PythonRunner::NumpyArray arr;
};

}

class HashSlotFilter
    : public SimpleUdf<std::string,
                       std::string,
                       std::vector<std::string>,
                       std::vector<std::string>,
                       std::vector<Tensor>,
                       std::string,
                       size_t,
                       size_t*> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const std::string& func_def,
      const std::string& func_name,
      const std::vector<std::string>& func_args,
      const std::vector<std::string>& payload_name,
      const std::vector<Tensor>& payload,
      const std::string& slot_name,
      const size_t& slot_size,
      size_t* del_size) const {
    PythonRunner runner;
    Variable* var = ctx->GetVariable();
    PS_CHECK_STATUS(runner.Init(func_def, func_name));
    std::unique_ptr<HashMap>& hashmap = (dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(var->GetSlicer()))->Internal();
    if (hashmap == nullptr) {
      return Status::ArgumentError("HashSlotFilter: Variable Should be a Hash Variable for " + ctx->GetVariableName());
    }
    Tensor* slot = var->GetVariableLikeSlot(slot_name, DataType::kFloat, TensorShape({slot_size}), []{ return new initializer::ConstantInitializer(0); });
    std::vector<Argument> arguments;
    for (auto&& arg : func_args) {
      Argument argument;
      auto iter = std::find(payload_name.begin(), payload_name.end(), arg);
      if (iter != payload_name.end()) {
        argument.type = Argument::kArr;
        argument.t = payload[iter - payload_name.begin()];
        PS_CHECK_STATUS(PythonRunner::ParseTensor(argument.t, &argument.arr));
      } else if (arg == "data_") {
        argument.type = Argument::kSlice;
        argument.t = *var->GetData();
      } else {
        Tensor* t;
        PS_CHECK_STATUS(var->GetExistSlot(arg, &t));
        argument.type = Argument::kSlice;
        argument.t = *t;
      }
      arguments.push_back(argument);
    }
    // Block Everything
    ctx->GetServerLocker()->ChangeType(QRWLocker::kWrite);
    size_t size = hashmap->GetSize();
    size_t segment_size = var->GetData()->SegmentSize();
    size_t segment_count = (size + segment_size - 1) / segment_size;
    *del_size = 0;
    for (size_t i = 0; i < segment_count; i++) {
      std::vector<PythonRunner::NumpyArray> real_args;
      for (auto& arg : arguments) {
        PythonRunner::NumpyArray real_arg;
        if (arg.type == Argument::kArr) {
          real_arg = arg.arr;
        } else {
          PS_CHECK_STATUS(PythonRunner::ParseSubTensor(arg.t, i, size, &real_arg));
        }
        real_args.push_back(real_arg);
      }
      PythonRunner::NumpyArray result, new_slot;
      PS_CHECK_STATUS(runner.Run(real_args, &result, &new_slot));
      if (result.shape.Size() != 1) {
        return Status::ArgumentError("HashSlotFilter: return array should be 1-D");
      }
      if (result.type != DataType::kInt8) {
        return Status::ArgumentError("HashSlotFilter: return array type should be Bool");
      }
      if (new_slot.shape.Size() != 2) {
        return Status::ArgumentError("HashSlotFilter: return slot array should be 2-D");
      }
      if (new_slot.type != DataType::kFloat) {
        return Status::ArgumentError("HashSlotFilter: return slot array type should be Float");
      }
      if (new_slot.shape[1] != slot_size) {
        return Status::ArgumentError("HashSlotFilter: return slot array should be [*, slot_size]");
      }
      std::vector<size_t> ids;
      for (size_t j = 0; j < result.shape[0]; j++) {
        if (((uint8_t*)result.data)[j]) {
          ids.push_back(j + segment_size * i);
        }
      }
      tbb::concurrent_vector<size_t> unfiltered_ids;
      *del_size += hashmap->EraseById(ctx->GetVariableName(), ids, &unfiltered_ids);
      for (auto id : unfiltered_ids) {
        memcpy(slot->Raw<float>(id), ((float*)new_slot.data) + slot_size * (id - segment_size * i), sizeof(float) * slot_size);
      }
    }
    ctx->GetServerLocker()->ChangeType(QRWLocker::kSimpleRead);
    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(HashSlotFilter, HashSlotFilter);

}
}
}

