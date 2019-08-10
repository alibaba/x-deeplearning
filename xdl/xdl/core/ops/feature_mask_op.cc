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
#include "xdl/core/utils/string_utils.h"

namespace xdl {

class FeatureMaskOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string mask_conf;
    XDL_CHECK_STATUS(ctx->GetAttr("mask_conf", &mask_conf));    
    XDL_CHECK_STATUS(ctx->GetAttr("index", &index_));    
    std::vector<std::string> fields = StringUtils::split(mask_conf, ";");
    for (auto& field: fields) {
      std::vector<std::string> kv = StringUtils::split(field, ":");
      XDL_CHECK_COND(kv.size() == 2, Status::ArgumentError("mask conf error"));
      std::vector<std::string> vlist = StringUtils::split(kv[1], ",");
      XDL_CHECK_COND(vlist.size() > 0, Status::ArgumentError("mask conf error"));
      int32_t tag = atoi(kv[0].c_str());
      for (auto& v: vlist) {
        tag_2_mask_[tag].push_back(atoi(v.c_str()));
      }
    }

    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input;
    XDL_CHECK_STATUS(ctx->GetInput("input", &input));
    Tensor tag_tensor;
    XDL_CHECK_STATUS(ctx->GetInput("tag", &tag_tensor));    
    XDL_CHECK_COND(
        tag_tensor.Shape().IsScalar(), 
        Status::ArgumentError("tag must be scalar"));
    int32_t tag = tag_tensor.Scalar<int32_t>();
    bool need_mask = false;
    auto it = tag_2_mask_.find(tag);
    if(it != tag_2_mask_.end()) {
      for (auto index: it->second) {
        if (index == index_) {
          need_mask = true;
          break;
        }
      }
    }

    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput("output", input.Shape(), &output));
    if (!need_mask) {
      memcpy(output.Raw<char*>(), 
             input.Raw<char*>(), 
             output.Shape().NumElements() * SizeOfType(output.Type()));
    } else {
      memset(output.Raw<char*>(), 
             0, 
             output.Shape().NumElements() * SizeOfType(output.Type()));
    }

    return Status::Ok();
  }

 private:
  std::unordered_map<int32_t, std::vector<int32_t> > tag_2_mask_;
  int64_t index_;
};

XDL_DEFINE_OP(FeatureMaskOp)
  .Input("input", DataType::kFloat)
  .Input("tag", DataType::kInt32)
  .Output("output", DataType::kFloat)
  .Attr("index", AttrValue::kInt)
  .Attr("mask_conf", AttrValue::kString);

XDL_REGISTER_KERNEL(FeatureMaskOp, FeatureMaskOp)
  .Device("CPU");

} // namespace xdl

