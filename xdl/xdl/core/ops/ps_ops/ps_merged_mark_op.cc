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

#include "ps-plus/client/partitioner/merged_broadcast.h"
#include "ps-plus/client/partitioner/merged_hash.h"
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

class PsMergedMarkOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string var_name_str;
    XDL_CHECK_STATUS(ctx->GetAttr("var_names", &var_name_str));
    var_names_ = StringUtils::split(var_name_str, ",");

    std::string pattern_str;
    XDL_CHECK_STATUS(ctx->GetAttr("patterns", &pattern_str));
    patterns_ = StringUtils::split(pattern_str, ",");
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);

    std::vector<Tensor> raw_ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("ids", &raw_ids), done);
    std::vector<Tensor> raw_i;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("i", &raw_i), done);
    std::vector<int64_t> i_vec;
    std::vector<float> save_ratio_vec;
    for (size_t i = 0; i < raw_i.size(); ++i) {
      i_vec.push_back(raw_i[i].Scalar<int64_t>());
      save_ratio_vec.push_back(0.0);
    }
       
    std::vector<ps::Tensor> id_vec;
    for (auto& id: raw_ids) {
      id_vec.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
          XDL2PS::ConvertTensorZC(id, &id_vec.back()),
          done);
    }

    std::vector<std::vector<std::unique_ptr<ps::Data>>>* outputs = 
      new std::vector<std::vector<std::unique_ptr<ps::Data>>>;
    auto cb = [ctx, outputs, done](const ps::Status& st) {
      delete outputs;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    ps::client::UdfData slice_udf("BuildHashSlice", 
                                  ps::client::UdfData(0), 
                                  ps::client::UdfData(3),
                                  ps::client::UdfData(4),                                  
                                  ps::client::UdfData(5),
                                  ps::client::UdfData(6));
    ps::client::UdfData udf("ScalarIntegerLogger", 
                            slice_udf, 
                            ps::client::UdfData(1), 
                            ps::client::UdfData(2));
    std::vector<ps::client::MergedPartitioner*> spliters{
        new ps::client::partitioner::MergedHashId,
        new ps::client::partitioner::MergedBroadcast,
        new ps::client::partitioner::MergedBroadcast,
        new ps::client::partitioner::MergedBroadcast,
        new ps::client::partitioner::MergedBroadcast,
        new ps::client::partitioner::MergedBroadcast,
        new ps::client::partitioner::MergedBroadcast};
    client->Process(
        udf, var_names_, 
        client->Args(id_vec, patterns_, i_vec, var_names_, save_ratio_vec, true, false),
        spliters, {}, outputs, cb);
  }

 private:
  std::vector<std::string> var_names_;
  std::vector<std::string> patterns_;
};

XDL_DEFINE_OP(PsMergedMarkOp)
  .InputListV2("ids", "input_type_0")
  .InputListV2("i", "input_type_1")
  .Attr("input_type_0", AttrValue::kDataTypeList)
  .Attr("input_type_1", AttrValue::kDataTypeList)
  .Attr("var_names", AttrValue::kString)
  .Attr("patterns", AttrValue::kString);

XDL_REGISTER_KERNEL(PsMergedMarkOp, PsMergedMarkOp).Device("CPU");

} // namespace xdl

