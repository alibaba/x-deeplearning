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

class PsSparsePullOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &ids), done);
    Tensor t_save_ratio;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_save_ratio), done);
    float save_ratio = t_save_ratio.Scalar<float>();
    ps::Tensor convert_ids;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(ids, &convert_ids),
        done);
    ps::Tensor* result = new ps::Tensor;
    auto cb = [result, ctx, done](const ps::Status& st) {
      std::unique_ptr<ps::Tensor> result_deleter(result);
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      XDL_CHECK_STATUS_ASYNC(
          PS2XDL::ConvertTensorWithCtx(*result, ctx, 0), 
          done);
      done(Status::Ok());
    };

    switch(var_type_) {
    case VarType::kIndex:
      client->SparsePull(var_name_, convert_ids, result, cb);
      break;
    case VarType::kHash128:
    case VarType::kHash64:
      client->HashPull(var_name_, convert_ids, save_ratio, result, cb);
      break;      
    default:
      XDL_CHECK_COND_ASYNC(
          false, 
          Status::ArgumentError("unsupported vartype"),
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
};

XDL_DEFINE_OP(PsSparsePullOp)
  .Input("ids", "dtype")
  .Input("save_ratio", DataType::kFloat)
  .Output("output", "otype")
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("otype", AttrValue::kDataType);

#define REGISTER_KERNEL(ITYPE, OTYPE) \
  XDL_REGISTER_KERNEL(PsSparsePullOp, PsSparsePullOp)  \
  .Device("CPU")                                       \
  .AttrDataType<ITYPE>("dtype")                        \
  .AttrDataType<OTYPE>("otype");                       \

REGISTER_KERNEL(int32_t, int8_t);
REGISTER_KERNEL(int32_t, int16_t);
REGISTER_KERNEL(int32_t, int32_t);
REGISTER_KERNEL(int32_t, int64_t);
REGISTER_KERNEL(int32_t, float);
REGISTER_KERNEL(int64_t, int8_t);
REGISTER_KERNEL(int64_t, int16_t);
REGISTER_KERNEL(int64_t, int32_t);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, float);

} // namespace xdl


