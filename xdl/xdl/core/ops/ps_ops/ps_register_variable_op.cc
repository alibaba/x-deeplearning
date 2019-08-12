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

#include "ps-plus/message/variable_info.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsRegisterVariableOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    XDL_CHECK_STATUS(ctx->GetAttr("shape", &shape_));    
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));    
    XDL_CHECK_STATUS(ctx->GetAttr("extra_info", &extra_info_));    
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS(GetClient(&client));
    ps::VariableInfo vi;
    vi.name = var_name_;
    for (auto dim: shape_.Dims()) {
      vi.shape.push_back(dim);
    }

    XDL_CHECK_STATUS(XDL2PS::ConvertType(dtype_, &vi.datatype));
    std::vector<std::string> arg_list = 
      StringUtils::split(extra_info_, ";");
    for (auto& item: arg_list) {
      std::vector<std::string> kv = StringUtils::split(item, "=");
      XDL_CHECK_COND(
          kv.size() == 2, 
          Status::ArgumentError("invalid arg:" + item));
      vi.args[kv[0]] = kv[1];
    }

    switch (var_type_) {
    case VarType::kIndex:
      vi.type = ps::VariableInfo::kIndex;
      break;
    case VarType::kHash128:
      vi.type = ps::VariableInfo::kHash128;
      break;
    case VarType::kHash64:
      vi.type = ps::VariableInfo::kHash64;
      break;
    default:
      XDL_CHECK_COND(
          false, 
          Status::ArgumentError("variable type error"));
    }

    XDL_CHECK_STATUS(
        PS2XDL::ConvertStatus(
            client->RegisterVariable(var_name_, vi)));
    return Status::Ok();
  }

 private:
  std::string var_name_;
  VarType var_type_;
  TensorShape shape_;
  DataType dtype_;
  std::string extra_info_;
};

XDL_DEFINE_OP(PsRegisterVariableOp)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("extra_info", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("shape", AttrValue::kTensorShape);

XDL_REGISTER_KERNEL(PsRegisterVariableOp, 
                    PsRegisterVariableOp).Device("CPU");

} // namespace xdl


