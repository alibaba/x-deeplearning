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
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

class ReadContextOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("ctx_id", &id_));
    return Status::Ok();
  }
  
  Status Compute(OpKernelContext* ctx) override {
    // TODO: Check Error
    ExecutorContext* ectx = ctx->GetRunOption().in_ctx;
    ctx->SetOutput(0, ectx->tensors[id_]);
    return Status::Ok();
  }
 private:
  int64_t id_;
};

XDL_DEFINE_OP(ReadContextOp)
  .Output("input", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("ctx_id", AttrValue::kInt);

XDL_REGISTER_KERNEL(ReadContextOp, ReadContextOp).Device("CPU");

class WriteContextOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("ctx_id", &id_));
    return Status::Ok();
  }
  
  Status Compute(OpKernelContext* ctx) override {
    // TODO: Check Error
    ExecutorContext* ectx = ctx->GetRunOption().out_ctx;
    if (id_ >= ectx->tensors.size()) {
      ectx->tensors.resize(id_ + 1);
    }
    ctx->GetInput(0, &ectx->tensors[id_]);
    return Status::Ok();
  }
 private:
  int64_t id_;
};

XDL_DEFINE_OP(WriteContextOp)
  .Input("input", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("ctx_id", AttrValue::kInt);

XDL_REGISTER_KERNEL(WriteContextOp, WriteContextOp).Device("CPU");


}  // namespace xdl

