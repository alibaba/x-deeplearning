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

class PsSparseStatisOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
  }
};

XDL_DEFINE_OP(PsSparseStatisOp)
  .Input("ids", "dtype")
  .Input("indexs", DataType::kInt32)
  .Input("segments", DataType::kInt32)
  .Input("sample_indexs", DataType::kInt32)
  .Input("sample_segments", DataType::kInt32)
  .Input("labels", DataType::kFloat)
  .Input("save_ratio", DataType::kFloat)
  .Input("global_step", DataType::kInt64)
  .Input("statis_decay", DataType::kDouble)
  .Input("statis_decay_period", DataType::kInt64)
  .Output("output", "otype")
  .Attr("statis_type", AttrValue::kString)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("otype", AttrValue::kDataType);

#define REGISTER_KERNEL(ITYPE, OTYPE) \
  XDL_REGISTER_KERNEL(PsSparseStatisOp, PsSparseStatisOp)  \
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


