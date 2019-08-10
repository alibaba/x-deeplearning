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

#include "xdl/core/ops/ps_ops/model_server/model_server_forward.h"

namespace xdl {

class PsModelServerForwardWaitOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("handle", &handle_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    PsModelServerForwardQueue::Instance()->Wait(handle_, [ctx, done](){
      done(Status::Ok());
    });
  }

 private:
  int64_t handle_;
};

XDL_DEFINE_OP(PsModelServerForwardWaitOp)
  .Attr("handle", AttrValue::kInt);

XDL_REGISTER_KERNEL(PsModelServerForwardWaitOp, PsModelServerForwardWaitOp)
  .Device("CPU");

class PsModelServerForwardRequestOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("handle", &handle_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    PsModelServerForwardQueue::Instance()->Pop(handle_, [ctx, done](PsModelServerForwardItem* rst){
      rst->run = false;
      ctx->AddDoneHandler([rst](Status st){
        if (!rst->run) {
          if (st.IsOk()) {
            st = Status::ArgumentError("model server has no output op");
          }
          rst->err(st);
        }
        delete rst;
      });
      Tensor output_handle;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertTensorWithCtx(rst->ids, ctx, 1), done);
      XDL_CHECK_STATUS_ASYNC(ctx->AllocateOutput(0, TensorShape(), &output_handle), done);
      output_handle.Raw<int64_t>()[0] = reinterpret_cast<int64_t>(rst);
      done(Status::Ok());
    });
  }

 private:
  int64_t handle_;
};

XDL_DEFINE_OP(PsModelServerForwardRequestOp)
  .Attr("handle", AttrValue::kInt)
  .Output("step_handle", DataType::kInt64)
  .Output("output_ids", DataType::kInt64);

XDL_REGISTER_KERNEL(PsModelServerForwardRequestOp, PsModelServerForwardRequestOp)
  .Device("CPU");

class PsModelServerForwardResponseOp : public xdl::OpKernel {
 public:
  Status Compute(OpKernelContext* ctx) override {
    Tensor handle, grads;
    XDL_CHECK_STATUS(ctx->GetInput(0, &handle));
    XDL_CHECK_STATUS(ctx->GetInput(1, &grads));
    XDL_CHECK_COND(handle.Shape().Dims().size() == 0 && handle.Type() == DataType::kInt64,
                   Status::ArgumentError("handle should be int64 scalar"));
    PsModelServerForwardItem* rst = reinterpret_cast<PsModelServerForwardItem*>(handle.Raw<int64_t>()[0]);
    ps::Tensor converted_result;
    XDL_CHECK_STATUS(XDL2PS::ConvertTensor(grads, &converted_result));
    rst->ok(converted_result);
    rst->run = true;
    return Status::Ok();
  }
};

XDL_DEFINE_OP(PsModelServerForwardResponseOp)
  .Attr("dtype", AttrValue::kDataType)
  .Input("step_handle", DataType::kInt64)
  .Input("grads", "dtype");

XDL_REGISTER_KERNEL(PsModelServerForwardResponseOp, PsModelServerForwardResponseOp)
  .Device("CPU");

} // namespace xdl


