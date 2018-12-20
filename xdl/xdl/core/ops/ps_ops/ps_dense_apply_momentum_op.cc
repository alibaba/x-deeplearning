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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsDenseApplyMomentumOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("use_nesterov", &use_nesterov_));
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_lr), done);
    double lr = t_lr.Scalar<double>();
    Tensor t_momentum;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_momentum), done);
    double momentum = t_momentum.Scalar<double>();
    Tensor grad;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(2, &grad), done);
    ps::Tensor convert_grad;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(grad, &convert_grad), 
        done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    switch (var_type_) {
    case VarType::kIndex:
      client->DensePush(
          var_name_, 
          "MomentumUpdater", 
          client->Args(convert_grad, lr, momentum, use_nesterov_), 
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
  bool use_nesterov_;
};

XDL_DEFINE_OP(PsDenseApplyMomentumOp)
  .Input("learning_rate", DataType::kDouble)
  .Input("momentum", DataType::kDouble)
  .Input("grad", DataType::kFloat)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("use_nesterov", AttrValue::kBool);

XDL_REGISTER_KERNEL(PsDenseApplyMomentumOp, PsDenseApplyMomentumOp)
  .Device("CPU");

} // namespace xdl


