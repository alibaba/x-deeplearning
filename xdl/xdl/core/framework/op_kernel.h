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

#ifndef XDL_CORE_FRAMEWORK_OP_KERNEL_H_
#define XDL_CORE_FRAMEWORK_OP_KERNEL_H_

#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/device.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/run_option.h"

namespace xdl {

struct OpKernelConstruction;
struct OpKernelContext;
class SimpleExecutor;

class OpKernelBase {
 public:
  using Callback = std::function<void(Status)>;
  virtual ~OpKernelBase() {}

  virtual Status Init(OpKernelConstruction* ctx);

  virtual void Launch(OpKernelContext* ctx) = 0;
};

class OpKernel : public OpKernelBase {
 public:
  void Launch(OpKernelContext* ctx) override;
  virtual Status Compute(OpKernelContext* ctx) = 0;
};

class OpKernelAsync : public OpKernelBase {
 public:
  void Launch(OpKernelContext* ctx) override;
  virtual void Compute(OpKernelContext* ctx, Callback done) = 0;
};

class OpKernelConstruction {
 public:
  explicit OpKernelConstruction(
      const std::unordered_map<std::string, AttrValue>& attr,
      Device* device)
    : attr_(attr), device_(device) {}
  Status GetAttr(const std::string& name, int64_t* i);
  Status GetAttr(const std::string& name, float* f);
  Status GetAttr(const std::string& name, bool* b);
  Status GetAttr(const std::string& name, std::string* s);
  Status GetAttr(const std::string& name, DataType* type);
  Status GetAttr(const std::string& name, TensorShape* shape);
  Status GetAttr(const std::string& name, std::vector<DataType>* type_list);
  Device* GetDevice();
 private:
  std::unordered_map<std::string, AttrValue> attr_;
  Device* device_;
};

struct OpKernelContextArg {
  Device* device;
  std::vector<Device*> input_devices;
  std::vector<std::string> input_name;
  std::vector<DataType> input_type;
  std::unordered_map<std::string, int> input_id;
  std::unordered_map<std::string, std::vector<int>> input_list_id;
  std::vector<std::string> output_name;
  std::vector<DataType> output_type;
  std::unordered_map<std::string, int> output_id;
  std::unordered_map<std::string, std::vector<int>> output_list_id;
};

class OpKernelContext : public RefCounted {
 public:
  OpKernelContext(OpKernelContextArg* arg,
                  SimpleExecutor* executor,
                  const std::vector<Tensor>& input);
  OpKernelContext(OpKernelContextArg* arg,
                  SimpleExecutor* executor,
                  std::vector<Tensor>&& input);
  Status GetInput(size_t id, Tensor* tensor);
  Status GetInput(const std::string& name, Tensor* tensor);
  Status GetInputList(const std::string& name,
                      std::vector<Tensor>* tensor_list);
  Status Allocate(const TensorShape& shape, DataType type, Tensor* tensor);
  Status AllocateOutput(size_t id, const TensorShape& shape, Tensor* tensor);
  Status AllocateOutput(const std::string& name, const TensorShape& shape,
                        Tensor* tensor);
  Status SetOutput(size_t id, const Tensor& tensor);
  Status SetOutput(const std::string& name, const Tensor& tensor);
  Status SetOutputList(const std::string& name,
                       const std::vector<Tensor>& tensor_list);
  void AddDoneHandler(std::function<void(Status)> handler);
  Device* GetDevice();
  void LaunchDone(Status st);
  void RunDone(Status st);
  const RunOption& GetRunOption();

  void SetLaunchDone(OpKernelBase::Callback launch_done);
  void SetRunDone(OpKernelBase::Callback run_done);
  const std::vector<Tensor>& GetOutputs() const {
    return output_;
  }

 private:
  OpKernelContextArg* arg_;
  SimpleExecutor* executor_;
  OpKernelBase::Callback launch_done_;
  OpKernelBase::Callback run_done_;

  std::vector<Tensor> allocated_;
  std::vector<Tensor> input_;
  std::vector<Tensor> output_;
};


#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK) \
  do {                                              \
    xdl::Status _s_(STATUS);                        \
    if (!_s_.IsOk()) {                              \
      CALLBACK(_s_);                                \
      return;                                       \
    }                                               \
  } while (0)                                       \

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_OP_KERNEL_H_

