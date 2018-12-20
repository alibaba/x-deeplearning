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

class PsDenseApplyAdamOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("lr_decay", &lr_decay_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_beta1;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_beta1), done);
    double beta1 = t_beta1.Scalar<double>();
    Tensor t_beta2;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_beta2), done);
    double beta2 = t_beta2.Scalar<double>();
    Tensor t_epsilon;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(2, &t_epsilon), done);
    double epsilon = t_epsilon.Scalar<double>();
    Tensor t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(3, &t_lr), done);
    double lr = t_lr.Scalar<double>();
    Tensor grad;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(4, &grad), done);
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
          "AdamUpdater", 
          client->Args(convert_grad, lr, epsilon, beta1, beta2, lr_decay_), 
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
  bool lr_decay_;
};

XDL_DEFINE_OP(PsDenseApplyAdamOp)
  .Input("beta1", DataType::kDouble)
  .Input("beta2", DataType::kDouble)
  .Input("epsilon", DataType::kDouble)
  .Input("learning_rate", DataType::kDouble)
  .Input("grad", DataType::kFloat)
  .Attr("var_name", AttrValue::kString)
  .Attr("lr_decay", AttrValue::kBool)
  .Attr("var_type", AttrValue::kString);

XDL_REGISTER_KERNEL(PsDenseApplyAdamOp, PsDenseApplyAdamOp)
  .Device("CPU");

} // namespace xdl
