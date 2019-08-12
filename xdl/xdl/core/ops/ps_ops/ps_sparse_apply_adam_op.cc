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

template <typename T>
class PsSparseApplyAdamOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
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
    Tensor t_lr_decay;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(4, &t_lr_decay), done);
    bool lr_decay = t_lr_decay.Scalar<bool>();
    Tensor grad;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(5, &grad), done);
    Tensor indices;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(6, &indices), done);    
    ps::Tensor convert_grad;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(grad, &convert_grad),
        done);
    ps::Tensor convert_indices;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(indices, &convert_indices),
        done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    std::vector<ps::Tensor> grad_vec = {convert_grad};
    std::vector<double> lr_vec = {lr};
    std::vector<double> epsilon_vec = {epsilon};
    std::vector<double> beta1_vec = {beta1};
    std::vector<double> beta2_vec = {beta2};
    std::vector<bool> lr_decay_vec = {lr_decay};

    switch(var_type_) {
    case VarType::kIndex:
      client->SparsePush(
          var_name_, 
          convert_indices, 
          "AdamUpdater", 
          client->Args(grad_vec, lr_vec, epsilon_vec, beta1_vec, beta2_vec, lr_decay_vec),
          cb);
      break;
    case VarType::kHash128:
    case VarType::kHash64:
      client->HashPush(
          var_name_, 
          convert_indices,
          0.0,
          false,
          "AdamUpdater",
          client->Args(grad_vec, lr_vec, epsilon_vec, beta1_vec, beta2_vec, lr_decay_vec),
          cb);      
      break;
    default:
      XDL_CHECK_COND_ASYNC(
          false, 
          Status::ArgumentError("unsupported vartype"),
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsSparseApplyAdamOp)
  .Input("beta1", DataType::kDouble)
  .Input("beta2", DataType::kDouble)
  .Input("epsilon", DataType::kDouble)
  .Input("learning_rate", DataType::kDouble)
  .Input("lr_decay", DataType::kBool)
  .Input("grad", DataType::kFloat)
  .Input("indices", "dtype")
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType);

DEFINE_INT_OP(XDL_REGISTER_KERNEL(PsSparseApplyAdamOp, PsSparseApplyAdamOp<T>)
              .Device("CPU")
              .AttrDataType<T>("dtype"))

} // namespace xdl
