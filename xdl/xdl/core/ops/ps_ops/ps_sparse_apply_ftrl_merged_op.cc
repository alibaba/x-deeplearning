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

class PsSparseApplyFtrlMergedOp : public xdl::OpKernelAsync {
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
    std::vector<Tensor> t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("learning_rate", &t_lr), done);
    std::vector<double> lr;
    for (size_t i = 0; i < t_lr.size(); ++i) {
      lr.push_back(t_lr[i].Scalar<double>());
    }
    std::vector<Tensor> t_lr_power;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("learning_rate_power", &t_lr_power), done);
    std::vector<double> lr_power;
    for (size_t i = 0; i < t_lr_power.size(); ++i) {
      lr_power.push_back(t_lr_power[i].Scalar<double>());
    }
    std::vector<Tensor> t_init_acc;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("initial_accumulator_value", &t_init_acc), done);
    std::vector<double> init_acc;
    for (size_t i = 0; i < t_init_acc.size(); ++i) {
      init_acc.push_back(t_init_acc[i].Scalar<double>());
    }
    std::vector<Tensor> t_l1_reg;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("l1_reg", &t_l1_reg), done);
    std::vector<double> l1_reg;
    for (size_t i = 0; i < t_l1_reg.size(); ++i) {
      l1_reg.push_back(t_l1_reg[i].Scalar<double>());
    }
    std::vector<Tensor> t_l2_reg;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("l2_reg", &t_l2_reg), done);
    std::vector<double> l2_reg;
    for (size_t i = 0; i < t_l2_reg.size(); ++i) {
      l2_reg.push_back(t_l2_reg[i].Scalar<double>());
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
                             "FtrlUpdater", 
                             client->Args(convert_grad, lr, lr_power, init_acc, l1_reg, l2_reg), 
                             cb);
    } else {
      done(Status::ArgumentError("PsSparseApplyFtrlMergedOp var_type must be hash"));
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
  std::vector<std::string> var_names_;
};

XDL_DEFINE_OP(PsSparseApplyFtrlMergedOp)
  .InputListV2("learning_rate", "input_type_0")
  .InputListV2("learning_rate_power", "input_type_1")
  .InputListV2("initial_accumulator_value", "input_type_2")
  .InputListV2("l1_reg", "input_type_3")
  .InputListV2("l2_reg", "input_type_4")
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

XDL_REGISTER_KERNEL(PsSparseApplyFtrlMergedOp, PsSparseApplyFtrlMergedOp).Device("CPU");

} // namespace xdl
