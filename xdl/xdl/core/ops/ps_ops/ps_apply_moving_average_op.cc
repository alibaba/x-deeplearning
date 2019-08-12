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

class PsApplyMovingAverageOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("moment", &moment_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor value;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &value), done);
    ps::Tensor converted_value;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(value, &converted_value), 
        done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    std::vector<float> moment_vec = {moment_};
    std::vector<ps::Tensor> value_vec = {converted_value};

    client->DensePush(var_name_, 
                      "MovingAverageUpdater", 
                      client->Args(moment_vec, value_vec),
                      cb);
  }

 private:
  std::string var_name_;
  float moment_;
};

XDL_DEFINE_OP(PsApplyMovingAverageOp)
  .Input("value", DataType::kFloat)
  .Attr("var_name", AttrValue::kString)
  .Attr("moment", AttrValue::kFloat);

XDL_REGISTER_KERNEL(PsApplyMovingAverageOp, PsApplyMovingAverageOp).Device("CPU");

} // namespace xdl


