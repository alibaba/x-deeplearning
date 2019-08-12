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
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsDenseApplyAdaptiveRevisionPullOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_worker_cnt;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_worker_cnt), done);
    int64_t worker_cnt = t_worker_cnt.Scalar<int64_t>();
    Tensor t_worker_idx;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_worker_idx), done);
    int64_t worker_idx = t_worker_idx.Scalar<int64_t>();
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    switch (var_type_) {
    case VarType::kIndex:
      client->DensePush(
          var_name_, 
          "AdaptiveRevisionPullUpdater", 
          client->Args(worker_cnt, worker_idx),
          cb);
      break;
    default:
      XDL_CHECK_COND_ASYNC(
          false, 
          Status::ArgumentError("variable type error"), 
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsDenseApplyAdaptiveRevisionPullOp)
  .Input("worker_cnt", DataType::kInt64)
  .Input("worker_idx", DataType::kInt64)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString);

XDL_REGISTER_KERNEL(PsDenseApplyAdaptiveRevisionPullOp, 
                    PsDenseApplyAdaptiveRevisionPullOp)
  .Device("CPU");

} // namespace xdl


