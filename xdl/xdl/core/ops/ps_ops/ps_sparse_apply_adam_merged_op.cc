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
#include "xdl/core/utils/string_utils.h"

namespace xdl {

class PsSparseApplyAdamMergedOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    std::string var_name_str;
    XDL_CHECK_STATUS(ctx->GetAttr("var_names", &var_name_str));
    var_names_ = StringUtils::split(var_name_str, ",");
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    std::vector<Tensor> t_beta1;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("beta1", &t_beta1), done);
    std::vector<double> beta1;
    for (size_t i = 0; i < t_beta1.size(); ++i) {
      beta1.push_back(t_beta1[i].Scalar<double>());
    }
    std::vector<Tensor> t_beta2;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("beta2", &t_beta2), done);
    std::vector<double> beta2;
    for (size_t i = 0; i < t_beta2.size(); ++i) {
      beta2.push_back(t_beta2[i].Scalar<double>());
    }
    std::vector<Tensor> t_epsilon;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("epsilon", &t_epsilon), done);
    std::vector<double> epsilon;
    for (size_t i = 0; i < t_epsilon.size(); ++i) {
      epsilon.push_back(t_epsilon[i].Scalar<double>());
    }
    std::vector<Tensor> t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("learning_rate", &t_lr), done);
    std::vector<double> lr;
    for (size_t i = 0; i < t_lr.size(); ++i) {
      lr.push_back(t_lr[i].Scalar<double>());
    }
    std::vector<Tensor> t_lr_decay;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("lr_decay", &t_lr_decay), done);
    std::vector<bool> lr_decay;
    for (size_t i = 0; i < t_lr_decay.size(); ++i) {
      lr_decay.push_back(t_lr_decay[i].Scalar<bool>());
    }
    std::vector<Tensor> grads;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("grad", &grads), done);
    std::vector<Tensor> indices;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("indices", &indices), done);    
    std::vector<ps::Tensor> convert_grad;
    for (auto& grad : grads) {
      convert_grad.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(grad, &convert_grad.back()),
        done);
    }
    std::vector<ps::Tensor> convert_indices;
    for (auto& indice : indices) {
      convert_indices.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(indice, &convert_indices.back()),
        done);
    }
    auto cb = [grads, indices, ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    std::vector<float> save_ratios;
    for (size_t i = 0; i < var_names_.size(); i++) {
      save_ratios.push_back(0.0);
    }
    if (var_type_ == VarType::kHash128 || var_type_ == VarType::kHash64) {
      client->MergedHashPush(var_names_,
                             convert_indices,
                             save_ratios,
                             "AdamUpdater", 
                             client->Args(convert_grad, lr, epsilon, beta1, beta2, lr_decay), 
                             cb);
    } else {
      done(Status::ArgumentError("PsSparseApplyAdamMergedOp var_type must be hash"));
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
  std::vector<std::string> var_names_;
};

XDL_DEFINE_OP(PsSparseApplyAdamMergedOp)
  .InputListV2("beta1", "input_type_0")
  .InputListV2("beta2", "input_type_1")
  .InputListV2("epsilon", "input_type_2")
  .InputListV2("learning_rate", "input_type_3")
  .InputListV2("lr_decay", "input_type_4")
  .InputListV2("grad", "input_type_5")
  .InputListV2("indices", "input_type_6")
  .Attr("input_type_0", AttrValue::kDataTypeList)
  .Attr("input_type_1", AttrValue::kDataTypeList)
  .Attr("input_type_2", AttrValue::kDataTypeList)
  .Attr("input_type_3", AttrValue::kDataTypeList)
  .Attr("input_type_4", AttrValue::kDataTypeList)
  .Attr("input_type_5", AttrValue::kDataTypeList)
  .Attr("input_type_6", AttrValue::kDataTypeList)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_names", AttrValue::kString)
  .Attr("var_type", AttrValue::kString);

XDL_REGISTER_KERNEL(PsSparseApplyAdamMergedOp, PsSparseApplyAdamMergedOp).Device("CPU");

} // namespace xdl
