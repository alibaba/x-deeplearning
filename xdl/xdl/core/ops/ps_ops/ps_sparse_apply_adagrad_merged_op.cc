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

class PsSparseApplyAdagradMergedOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    std::string var_name_str;
    XDL_CHECK_STATUS(ctx->GetAttr("var_names", &var_name_str));
    var_names_ = StringUtils::split(var_name_str, ",");
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    std::vector<Tensor> t_lr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("learning_rate", &t_lr), done);
    std::vector<Tensor> t_init_acc;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("initial_accumulator_value", &t_init_acc), done);
    std::vector<Tensor> grads;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("grad", &grads), done);
    std::vector<Tensor> indices;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("indices", &indices), done);
    std::vector<double> lr_vec;
    for (size_t i = 0; i < t_lr.size(); ++i) {
      lr_vec.push_back(t_lr[i].Scalar<double>());
    }
    std::vector<double> acc_vec;
    for (size_t i = 0; i < t_init_acc.size(); ++i) {
      acc_vec.push_back(t_init_acc[i].Scalar<double>());
    }
    std::vector<ps::Tensor> grad_vec;
    for (auto& grad : grads) {
      grad_vec.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(grad, &grad_vec.back()),
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
                             "AdagradUpdater", 
                             client->Args(grad_vec, lr_vec, acc_vec), 
                             cb);
    } else {
      done(Status::ArgumentError("PsSparseApplyAdagradMergedOp var_type must be hash"));
    }
  }

 private:
  std::string var_name_;
  std::vector<std::string> var_names_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsSparseApplyAdagradMergedOp)
  .InputListV2("learning_rate", "input_type_0")
  .InputListV2("initial_accumulator_value", "input_type_1")
  .InputListV2("grad", "input_type_2")
  .InputListV2("indices", "input_type_3")
  .Attr("input_type_0", AttrValue::kDataTypeList)
  .Attr("input_type_1", AttrValue::kDataTypeList)
  .Attr("input_type_2", AttrValue::kDataTypeList)
  .Attr("input_type_3", AttrValue::kDataTypeList)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_names", AttrValue::kString)
  .Attr("var_type", AttrValue::kString);

XDL_REGISTER_KERNEL(PsSparseApplyAdagradMergedOp, PsSparseApplyAdagradMergedOp).Device("CPU");

} // namespace xdl


