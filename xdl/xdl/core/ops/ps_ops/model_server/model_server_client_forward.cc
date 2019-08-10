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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"

namespace xdl {

class PsModelServerClientForwardOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("server_type", &server_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &ids), done);
    ps::Tensor convert_ids;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(ids, &convert_ids),
        done);
    ps::Tensor* result = new ps::Tensor;
    auto cb = [result, ctx, done](const ps::Status& st) {
      std::unique_ptr<ps::Tensor> result_deleter(result);
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      XDL_CHECK_STATUS_ASYNC(
          PS2XDL::ConvertTensorWithCtx(*result, ctx, 0), 
          done);
      done(Status::Ok());
    };
    client->ModelServerForward(server_type_, convert_ids, result, cb);
  }

 private:
  int64_t server_type_;
};

XDL_DEFINE_OP(PsModelServerClientForwardOp)
  .Input("ids", DataType::kInt64)
  .Output("result", "dtype")
  .Attr("server_type", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType);

XDL_REGISTER_KERNEL(PsModelServerClientForwardOp, PsModelServerClientForwardOp)
              .Device("CPU");

}  // namespace xdl
