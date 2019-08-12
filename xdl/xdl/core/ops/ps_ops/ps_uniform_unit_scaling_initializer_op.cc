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
#include "ps-plus/common/initializer/uniform_unit_scaling_initializer.h"

namespace xdl {

class PsUniformUnitScalingInitializerOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("seed", &seed_));
    XDL_CHECK_STATUS(ctx->GetAttr("factor", &factor_));
    XDL_CHECK_STATUS(ctx->GetAttr("shape", &shape_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    auto cb = [ctx, done](const ps::Status& st) {
      if (st.Code() == ps::Status::kAlreadyExist) {
        done(Status::Ok());
        return;
      }

      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    ps::TensorShape shape;
    XDL_CHECK_STATUS_ASYNC(XDL2PS::ConvertShape(shape_, &shape), done);
    ps::Initializer* initializer = 
      new ps::initializer::UniformUnitScalingInitializer(
          shape, seed_, factor_);
    switch (var_type_) {
    case VarType::kIndex:
      client->IndexInitializer(var_name_, initializer, cb);
      break;
    case VarType::kHash128:
    case VarType::kHash64:
      client->HashInitializer(var_name_, initializer, cb);
      break;              
    default:
      XDL_CHECK_COND_ASYNC(
          false,
          Status::ArgumentError("unsupported var type"),
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
  TensorShape shape_;
  int64_t seed_;
  float factor_;
};

XDL_DEFINE_OP(PsUniformUnitScalingInitializerOp)
  .Attr("var_name", AttrValue::kString)
  .Attr("seed", AttrValue::kInt)
  .Attr("factor", AttrValue::kFloat)
  .Attr("var_type", AttrValue::kString)
  .Attr("shape", AttrValue::kTensorShape);

XDL_REGISTER_KERNEL(
    PsUniformUnitScalingInitializerOp, 
    PsUniformUnitScalingInitializerOp).Device("CPU");

} // namespace xdl


