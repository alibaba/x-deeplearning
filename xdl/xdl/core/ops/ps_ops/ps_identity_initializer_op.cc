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

class PsIdentityInitializerOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor value;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &value), done);
    ps::Tensor convert_value;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(value, &convert_value), 
        done);
    auto cb = [ctx, done](const ps::Status& st) {
      if (st.Code() == ps::Status::kAlreadyExist) {
        done(Status::Ok());
        return;
      }

      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    switch (var_type_) {
    case VarType::kIndex:
      client->IdentityInitializer(var_name_, convert_value, cb);
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
};

XDL_DEFINE_OP(PsIdentityInitializerOp)
  .Input("value", "dtype")
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType);

XDL_REGISTER_KERNEL(PsIdentityInitializerOp, PsIdentityInitializerOp)
  .Device("CPU");

} // namespace xdl


