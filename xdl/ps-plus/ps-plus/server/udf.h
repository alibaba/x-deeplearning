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

#ifndef PS_SERVER_UDF_H_
#define PS_SERVER_UDF_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"
#include "ps-plus/common/plugin.h"
#include "ps-plus/server/udf_context.h"

#include <memory>
#include <vector>

namespace ps {
namespace server {

class StorageManager;

class Udf {
 public:
  virtual ~Udf() {}

  void Init(const std::vector<size_t>& inputs, const std::vector<size_t>& outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
  }

  virtual Status Run(UdfContext* ctx) const = 0;

 protected:
  Status GetInput(UdfContext* ctx, size_t id, Data** result) const {
    if(id >= inputs_.size()) {
      return Status::IndexOverflow("Udf Input Index Overflow.");
    };
    return ctx->GetData(inputs_[id], result);
  }

  Status SetOutput(UdfContext* ctx, size_t id, Data* output, bool need_free = true) const {
    if(id >= outputs_.size()) {
      if (need_free) {
        delete output;
      }
      return Status::IndexOverflow("Udf Output Index Overflow.");
    };
    return ctx->SetData(outputs_[id], output, need_free);
  }

  Status GetInputs(UdfContext* ctx, std::vector<Data*>* result) const {
    result->clear();
    for (auto id : inputs_) {
      Data* data;
      PS_CHECK_STATUS(ctx->GetData(id, &data));
      result->push_back(data);
    }
    return Status::Ok();
  }

  size_t InputSize() const {
    return inputs_.size();
  }

  size_t OutputSize() const {
    return outputs_.size();
  }

  Status AddDependency(UdfContext* ctx, Data* dependency) const {
    return ctx->AddDependency(dependency);
  }

  StorageManager* GetStorageManager(UdfContext* ctx) const {
    return ctx->GetStorageManager();
  }

  Variable* GetVariable(UdfContext* ctx) const {
    return ctx->GetVariable();
  }

  std::string GetVariableName(UdfContext* ctx) const {
    return ctx->GetVariableName();
  }

 private:
  std::vector<size_t> inputs_;
  std::vector<size_t> outputs_;
};

class UdfRegistry {
 public:
  virtual ~UdfRegistry() {}
  virtual Udf* Build(const std::vector<size_t>& inputs, const std::vector<size_t>& outputs) = 0;
  size_t InputSize() { return input_size_; }
  size_t OutputSize() { return output_size_; }
  static UdfRegistry* Get(const std::string& name) { return GetPlugin<UdfRegistry>(name); }
 protected:
  size_t input_size_;
  size_t output_size_;
};

template <typename T>
class UdfRegistryImpl : public UdfRegistry {
 public:
  UdfRegistryImpl(size_t input_size, size_t output_size) {
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual Udf* Build(const std::vector<size_t>& inputs, const std::vector<size_t>& outputs) {
    Udf* udf = new T;
    udf->Init(inputs, outputs);
    return udf;
  }
};

#define UDF_REGISTER(TYPE, NAME, INPUT, OUTPUT) \
  PLUGIN_REGISTER(ps::server::UdfRegistry, NAME, ps::server::UdfRegistryImpl<TYPE>, INPUT, OUTPUT)

}
}

#endif

