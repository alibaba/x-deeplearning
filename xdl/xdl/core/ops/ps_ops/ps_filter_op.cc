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

#include "ps-plus/client/partitioner/broadcast.h"
#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsFilterOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("pattern", &pattern_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_d;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_d), done);
    double d = t_d.Scalar<double>();
    Tensor t_i;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_i), done);
    int64_t i = t_i.Scalar<int64_t>();
    std::vector<std::unique_ptr<ps::Data>>* outputs = 
      new std::vector<std::unique_ptr<ps::Data>>;
    auto cb = [ctx, outputs, done](const ps::Status& st) {
      delete outputs;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };
    ps::client::UdfData udf("HashUnaryFilter", 
                            ps::client::UdfData(0), 
                            ps::client::UdfData(1), 
                            ps::client::UdfData(2));
    std::vector<ps::client::Partitioner*> spliters{
      new ps::client::partitioner::Broadcast, 
        new ps::client::partitioner::Broadcast,
        new ps::client::partitioner::Broadcast};
    client->Process(udf, var_name_, client->Args(pattern_, d, i), 
                    spliters, {}, outputs, cb);
  }

 private:
  std::string var_name_;
  std::string pattern_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsFilterOp)
  .Input("d", DataType::kDouble)
  .Input("i", DataType::kInt64)
  .Attr("var_name", AttrValue::kString)
  .Attr("pattern", AttrValue::kString);

XDL_REGISTER_KERNEL(PsFilterOp, PsFilterOp).Device("CPU");

} // namespace xdl


