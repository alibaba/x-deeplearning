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
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsMergedSparsePullOp : public xdl::OpKernelAsync {
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
    std::vector<Tensor> raw_ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("ids", &raw_ids), done);
    std::vector<Tensor> t_sr;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("save_ratios", &t_sr), done);
    std::vector<float> save_ratios;
    for (size_t i = 0; i < t_sr.size(); ++i) {
      save_ratios.push_back(t_sr[i].Scalar<float>());
    }
    
    
    std::vector<ps::Tensor> converted_ids;
    for (auto& id: raw_ids) {
      converted_ids.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
          XDL2PS::ConvertTensorZC(id, &converted_ids.back()),
          done);
    }

    std::vector<ps::Tensor>* ps_result = new std::vector<ps::Tensor>();
    auto cb = [raw_ids, ps_result, ctx, done](const ps::Status& st) {
      std::vector<Tensor> result;
      std::unique_ptr<std::vector<ps::Tensor> > result_deleter(ps_result);
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      std::chrono::time_point<std::chrono::system_clock> time_start, time_end;
      for (auto& item: *ps_result) {
        result.emplace_back();
        XDL_CHECK_STATUS_ASYNC(
            PS2XDL::ConvertTensorZC(item, &result.back()),
            done);
      }
      ctx->SetOutputList("outputs", result);
      done(Status::Ok());
    };

    if (var_type_ == VarType::kHash128 || var_type_ == VarType::kHash64) {
      client->MergedHashPull(var_names_, converted_ids, save_ratios, ps_result, cb);
    } else {
      done(Status::ArgumentError("PsMergedSparsePullOp var_type must be hash"));
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
  std::vector<std::string> var_names_;
};

XDL_DEFINE_OP(PsMergedSparsePullOp)
  .InputListV2("ids", "input_type_0")
  .InputListV2("save_ratios", "input_type_1")
  .OutputList("outputs", DataType::kFloat, "output_size")
  .Attr("input_type_0", AttrValue::kDataTypeList)
  .Attr("input_type_1", AttrValue::kDataTypeList)
  .Attr("output_size", AttrValue::kInt)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("var_names", AttrValue::kString);

XDL_REGISTER_KERNEL(PsMergedSparsePullOp, PsMergedSparsePullOp).Device("CPU");

} // namespace xdl


