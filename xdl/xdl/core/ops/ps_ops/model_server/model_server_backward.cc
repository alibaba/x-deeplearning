/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/ops/ps_ops/model_server/model_server_backward.h"

namespace xdl {

class PsModelServerBackwardWaitOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("handle", &handle_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    PsModelServerBackwardQueue::Instance()->Wait(handle_, [ctx, done](){
      done(Status::Ok());
    });
  }

 private:
  int64_t handle_;
};

XDL_DEFINE_OP(PsModelServerBackwardWaitOp)
  .Attr("handle", AttrValue::kInt);

XDL_REGISTER_KERNEL(PsModelServerBackwardWaitOp, PsModelServerBackwardWaitOp)
  .Device("CPU");

class PsModelServerBackwardRequestOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("handle", &handle_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    PsModelServerBackwardQueue::Instance()->Pop(handle_, [ctx, done](PsModelServerBackwardItem* rst){
      rst->run = false;
      ctx->AddDoneHandler([rst](Status st){
        if (!rst->run) {
          if (st.IsOk()) {
            st = Status::ArgumentError("model server has no output op");
          }
          rst->done(st);
        }
        delete rst;
      });
      Tensor output_handle;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertTensorWithCtx(rst->ids, ctx, 1), done);
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertTensorWithCtx(rst->grads, ctx, 2), done);
      XDL_CHECK_STATUS_ASYNC(ctx->AllocateOutput(0, TensorShape(), &output_handle), done);
      output_handle.Raw<int64_t>()[0] = reinterpret_cast<int64_t>(rst);
      done(Status::Ok());
    });
  }

 private:
  int64_t handle_;
};

XDL_DEFINE_OP(PsModelServerBackwardRequestOp)
  .Attr("handle", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Output("step_handle", DataType::kInt64)
  .Output("output_ids", DataType::kInt64)
  .Output("output_grads", "dtype");

XDL_REGISTER_KERNEL(PsModelServerBackwardRequestOp, PsModelServerBackwardRequestOp)
  .Device("CPU");

class PsModelServerBackwardResponseOp : public xdl::OpKernel {
 public:
  Status Compute(OpKernelContext* ctx) override {
    Tensor handle;
    XDL_CHECK_STATUS(ctx->GetInput(0, &handle));
    XDL_CHECK_COND(handle.Shape().Dims().size() == 0 && handle.Type() == DataType::kInt64,
                   Status::ArgumentError("handle should be int64 scalar"));
    PsModelServerBackwardItem* rst = reinterpret_cast<PsModelServerBackwardItem*>(handle.Raw<int64_t>()[0]);
    rst->done(Status::Ok());
    rst->run = true;
    return Status::Ok();
  }
};

XDL_DEFINE_OP(PsModelServerBackwardResponseOp)
  .Input("step_handle", DataType::kInt64);

XDL_REGISTER_KERNEL(PsModelServerBackwardResponseOp, PsModelServerBackwardResponseOp)
  .Device("CPU");

} // namespace xdl


