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

#include "ps-plus/client/partitioner/broadcast.h"
#include "ps-plus/client/partitioner/reduce.h"
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsSlotFilterOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("func_def", &func_def_));
    XDL_CHECK_STATUS(ctx->GetAttr("func_name", &func_name_));
    std::string func_args;
    XDL_CHECK_STATUS(ctx->GetAttr("func_args", &func_args));
    func_args_ = StringUtils::split(func_args, ";");
    std::string payload_name;
    XDL_CHECK_STATUS(ctx->GetAttr("payload_name", &payload_name));
    payload_name_ = StringUtils::split(payload_name, ";");
    XDL_CHECK_STATUS(ctx->GetAttr("slot_name", &slot_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("slot_size", &slot_size_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    std::vector<Tensor> payload_org;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("payload", &payload_org), done);
    std::vector<ps::Tensor> payload;
    for (auto& p : payload_org) {
      payload.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(p, &payload.back()),
        done);
    }
    ps::client::UdfData udf("HashSlotFilter", 
                            ps::client::UdfData(0), 
                            ps::client::UdfData(1), 
                            ps::client::UdfData(2),
                            ps::client::UdfData(3),
                            ps::client::UdfData(4),
                            ps::client::UdfData(5),
                            ps::client::UdfData(6)
                            );
    std::vector<ps::client::Partitioner*> spliters{
      new ps::client::partitioner::Broadcast, 
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast
    };
    std::vector<ps::client::Partitioner*> combiners{
      new ps::client::partitioner::ReduceSum<size_t>
    };
    std::vector<std::unique_ptr<ps::Data>>* outputs = 
      new std::vector<std::unique_ptr<ps::Data>>;    
    auto cb = [ctx, done, outputs](const ps::Status& st) {
      std::vector<std::unique_ptr<ps::Data>> o = std::move(*outputs);
      delete outputs;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      ps::WrapperData<size_t>* rst = dynamic_cast<ps::WrapperData<size_t>*>(o[0].get());
      if (rst == nullptr) {
        done(Status::ArgumentError("HashSimpleFilter Return Error"));
        return;
      }
      Tensor t;
      XDL_CHECK_STATUS_ASYNC(ctx->AllocateOutput(0, TensorShape({}), &t), done);
      t.Raw<int64_t>()[0] = rst->Internal();
      done(Status::Ok());
    };
    client->Process(udf, var_name_, client->Args(func_def_, func_name_, func_args_, payload_name_, payload, slot_name_, (size_t)slot_size_),
                    spliters, combiners, outputs, cb);
  }

 private:
  std::string var_name_;
  std::string func_def_;
  std::string func_name_;
  std::vector<std::string> func_args_;
  std::vector<std::string> payload_name_;
  std::string slot_name_;
  int64_t slot_size_;
};

XDL_DEFINE_OP(PsSlotFilterOp)
  .InputListV2("payload", "payload_type")
  .Attr("var_name", AttrValue::kString)
  .Attr("func_def", AttrValue::kString)
  .Attr("func_name", AttrValue::kString)
  .Attr("func_args", AttrValue::kString)
  .Attr("payload_name", AttrValue::kString)
  .Attr("payload_type", AttrValue::kDataTypeList)
  .Attr("slot_name", AttrValue::kString)
  .Attr("slot_size", AttrValue::kInt)
  .Output("del_size", DataType::kInt64);

XDL_REGISTER_KERNEL(PsSlotFilterOp, PsSlotFilterOp).Device("CPU");

} // namespace xdl


