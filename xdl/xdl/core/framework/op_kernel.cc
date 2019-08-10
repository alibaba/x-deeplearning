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

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/simple_executor.h"

#include <string>
#include <vector>

#include "xdl/core/utils/logging.h"

namespace xdl {

Status OpKernelBase::Init(OpKernelConstruction* ctx) {
  return Status::Ok();
}

void OpKernel::Launch(OpKernelContext* ctx) {
  Status status = Compute(ctx);
  ctx->LaunchDone(status);
  ctx->RunDone(Status::Ok());
}

void OpKernelAsync::Launch(OpKernelContext* ctx) {
  Callback done = [ctx](Status st){
    ctx->LaunchDone(st);
    ctx->RunDone(Status::Ok());
  };
  Compute(ctx, done);
}

Status OpKernelConstruction::GetAttr(const std::string& name,
                                     int64_t* i) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kInt,
                 Status::ArgumentError("Attr " + name + " is not int"));
  *i = iter->second.i;
  return Status::Ok();
}

Status OpKernelConstruction::GetAttr(const std::string& name,
                                     float* f) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kFloat,
                 Status::ArgumentError("Attr " + name + " is not float"));
  *f = iter->second.f;
  return Status::Ok();
}

Status OpKernelConstruction::GetAttr(const std::string& name,
                                     bool* b) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kBool,
                 Status::ArgumentError("Attr " + name + " is not bool"));
  *b = iter->second.b;
  return Status::Ok();
}

Status OpKernelConstruction::GetAttr(const std::string& name,
                                     std::string* s) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kString,
                 Status::ArgumentError("Attr " + name + " is not string"));
  *s = iter->second.s;
  return Status::Ok();
}

Status OpKernelConstruction::GetAttr(const std::string& name,
                                     DataType* type) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kDataType,
                 Status::ArgumentError("Attr " + name + " is not data type"));
  *type = iter->second.type;
  return Status::Ok();
}
Status OpKernelConstruction::GetAttr(const std::string& name,
                                     TensorShape* shape) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kTensorShape,
                 Status::ArgumentError("Attr " + name + " is not shape"));
  *shape = iter->second.shape;
  return Status::Ok();
}
Status OpKernelConstruction::GetAttr(const std::string& name,
                                     std::vector<DataType>* type_list) {
  auto iter = attr_.find(name);
  XDL_CHECK_COND(iter != attr_.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  XDL_CHECK_COND(iter->second.attr_type == AttrValue::kDataTypeList,
                 Status::ArgumentError("Attr " + name + " is not type_list"));
  *type_list = iter->second.type_list;
  return Status::Ok();
}

Device* OpKernelConstruction::GetDevice() {
  return device_;
}

OpKernelContext::OpKernelContext(OpKernelContextArg* arg,
                                 SimpleExecutor* executor,
                                 const std::vector<Tensor>& input)
  : RefCounted(2), arg_(arg), executor_(executor),
    input_(input), output_(arg->output_name.size()) {}

OpKernelContext::OpKernelContext(OpKernelContextArg* arg,
                                 SimpleExecutor* executor,
                                 std::vector<Tensor>&& input)
  : RefCounted(2), arg_(arg), executor_(executor),
    input_(), output_(arg->output_name.size()) {
  input_.swap(input);
}

Status OpKernelContext::GetInput(size_t id, Tensor* tensor) {
  XDL_CHECK_COND(id < input_.size(),
                 Status::ArgumentError("input id overflow"));
  *tensor = input_[id];
  return Status::Ok();
}

Status OpKernelContext::GetInput(const std::string& name, Tensor* tensor) {
  auto iter = arg_->input_id.find(name);
  XDL_CHECK_COND(iter != arg_->input_id.end(),
                 Status::ArgumentError("input name not found"));
  *tensor = input_[iter->second];
  return Status::Ok();
}

Status OpKernelContext::GetInputList(const std::string& name,
                                     std::vector<Tensor>* tensor_list) {
  auto iter = arg_->input_list_id.find(name);
  XDL_CHECK_COND(iter != arg_->input_list_id.end(),
                 Status::ArgumentError("input list name not found"));
  tensor_list->clear();
  for (auto item : iter->second) {
    tensor_list->push_back(input_[item]);
  }
  return Status::Ok();
}

Status OpKernelContext::Allocate(const TensorShape& shape,
                                 DataType type,
                                 Tensor* tensor) {
  *tensor = Tensor(arg_->device, shape, type);
  allocated_.push_back(*tensor);
  return Status::Ok();
}

Status OpKernelContext::AllocateOutput(size_t id,
                                       const TensorShape& shape,
                                       Tensor* tensor) {
  XDL_CHECK_COND(id < output_.size(),
                 Status::ArgumentError("output id overflow"));
  XDL_CHECK_STATUS(Allocate(shape, arg_->output_type[id], tensor));
  output_[id] = *tensor;
  return Status::Ok();
}

Status OpKernelContext::AllocateOutput(const std::string& name,
                                       const TensorShape& shape,
                                       Tensor* tensor) {
  auto iter = arg_->output_id.find(name);
  XDL_CHECK_COND(iter != arg_->output_id.end(),
                 Status::ArgumentError("output name not found"));
  int id = iter->second;
  XDL_CHECK_STATUS(Allocate(shape, arg_->output_type[id], tensor));
  output_[id] = *tensor;
  return Status::Ok();
}

Status OpKernelContext::SetOutput(size_t id, const Tensor& tensor) {
  XDL_CHECK_COND(id < output_.size(),
                 Status::ArgumentError("output id overflow"));
  output_[id] = tensor;
  return Status::Ok();
}

Status OpKernelContext::SetOutput(const std::string& name,
                                  const Tensor& tensor) {
  auto iter = arg_->output_id.find(name);
  XDL_CHECK_COND(iter != arg_->output_id.end(),
                 Status::ArgumentError("output name not found"));
  output_[iter->second] = tensor;
  return Status::Ok();
}

Status OpKernelContext::SetOutputList(const std::string& name,
                                      const std::vector<Tensor>& tensor_list) {
  auto iter = arg_->output_list_id.find(name);
  XDL_CHECK_COND(iter != arg_->output_list_id.end(),
                 Status::ArgumentError("output list name not found"));
  XDL_CHECK_COND(tensor_list.size() == iter->second.size(),
                 Status::ArgumentError("output list size not match"));
  for (size_t i = 0; i < tensor_list.size(); i++) {
    output_[iter->second[i]] = tensor_list[i];
  }
  return Status::Ok();
}

void OpKernelContext::AddDoneHandler(std::function<void(Status)> handler) {
  executor_->AddDoneHandler(handler);
}

Device* OpKernelContext::GetDevice() {
  return arg_->device;
}

void OpKernelContext::LaunchDone(Status st) {
  launch_done_(st);
}

void OpKernelContext::RunDone(Status st) {
  run_done_(st);
}

const RunOption& OpKernelContext::GetRunOption() {
  return executor_->GetRunOption();
}

void OpKernelContext::SetLaunchDone(OpKernelBase::Callback launch_done) {
  launch_done_ = launch_done;
}

void OpKernelContext::SetRunDone(OpKernelBase::Callback run_done) {
  run_done_ = run_done;
}

}  // namespace xdl

