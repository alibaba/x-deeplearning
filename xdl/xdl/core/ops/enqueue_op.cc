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

#include "xdl/core/utils/string_utils.h"
#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/tbb_concurrent_queue.h"

namespace xdl {

class EnqueueOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string names_str;
    XDL_CHECK_STATUS(ctx->GetAttr("names", &names_str));
    names_ = StringUtils::split(names_str, ";");
    TBBConcurrentQueue::Global()->Raw()->set_capacity(1000);
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    std::vector<Tensor> tensors;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("tensors", &tensors), done);
    XDL_CHECK_COND_ASYNC(tensors.size() == names_.size(), Status::ArgumentError("EnqueueOp: tensors and names size not match."), done);
    TBBConcurrentQueue::Global()->Raw()->push(tensors);
    done(Status::Ok());
  }

 private:
  std::vector<std::string> names_;
};

XDL_DEFINE_OP(EnqueueOp)
  .InputListV2("tensors", "types")
  .Attr("names", AttrValue::kString)
  .Attr("types", AttrValue::kDataTypeList);

XDL_REGISTER_KERNEL(EnqueueOp, EnqueueOp).Device("CPU");

} // namespace xdl


