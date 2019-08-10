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


#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/timer.h"
#include "xdl/data_io/data_io.h"

namespace xdl {

class ReleaseBatchOp: public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string ds;
    XDL_CHECK_STATUS(ctx->GetAttr("ds", &ds));

    data_io_ = io::DataIOMap::Instance()->Get(ds);
    XDL_CHECK(data_io_ != nullptr);

    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    data_io_->ReleaseBatch();
    return Status::Ok();
  }

 private:
  io::DataIO *data_io_;
};

XDL_DEFINE_OP(ReleaseBatch)
  .Attr("ds", AttrValue::kString);

XDL_REGISTER_KERNEL(ReleaseBatch, ReleaseBatchOp).Device("CPU");

}  // namespace xdl
