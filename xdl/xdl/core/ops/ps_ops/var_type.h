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

#ifndef XDL_CORE_OPS_PS_OPS_VAR_TYPE_H_
#define XDL_CORE_OPS_PS_OPS_VAR_TYPE_H_

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"

namespace xdl {

enum VarType {
  kIndex = 0,
  kHash128 = 1,
  kHash64 = 2,
};

inline Status XdlGetVarType(OpKernelConstruction* ctx, VarType* vtype) {
  std::string var_type;
  XDL_CHECK_STATUS(ctx->GetAttr("var_type", &var_type));
  if (var_type == "hash128" or var_type == "hash") {
    *vtype = VarType::kHash128;
  } else if (var_type == "hash64") {
    *vtype = VarType::kHash64;
  } else if (var_type == "index") {
    *vtype = VarType::kIndex;
  } else {
    return Status::ArgumentError("unsupported var type:" + var_type);
  }

  return Status::Ok();
}

} // namespace xdl

#endif // XDL_CORE_OPS_PS_OPS_VAR_TYPE_H_
