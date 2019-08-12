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
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/backend/device_singleton.h"
#include "xdl/core/backend/mxnet/mxnet_runner.h"
#include "xdl/core/backend/mxnet/mxnet_runner_holder.h"
#include "xdl/core/backend/mxnet/convert_utils.h"
#include "xdl/core/framework/gpu/gpu_device.h"

namespace xdl {

class MxnetBackendGpuOp : public xdl::GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) {
    int64_t id;
    XDL_CHECK_STATUS(ctx->GetAttr("id", &id));
    std::function<Status(MxnetRunnerHolder&)> init = 
    [ctx](MxnetRunnerHolder& holder)->Status{
      std::string var_name;
      XDL_CHECK_STATUS(ctx->GetAttr("var_name_str", &var_name));
      XDL_CHECK_STATUS(ctx->GetAttr("graph_def", &holder.graph_def_));
      XDL_CHECK_STATUS(ctx->GetAttr("device_type", &holder.device_type_));
      XDL_CHECK_STATUS(ctx->GetAttr("gradient_size", &holder.gradient_size_));
      XDL_CHECK_STATUS(ctx->GetAttr("is_training", &holder.is_training_));
      XDL_CHECK_STATUS(ctx->GetAttr("has_init_grad", &holder.has_init_grad_));
      XDL_CHECK_COND(holder.device_type_ == "gpu", 
                     Status::ArgumentError("device type must be gpu"));
      holder.var_names_ = StringUtils::split(var_name, ",");
      holder.mxnet_runner_.reset(new MxnetRunner(holder.is_training_));
      XDL_CHECK_STATUS(holder.mxnet_runner_->Init(holder.graph_def_, holder.device_type_));
      holder.context_.reset(new mxnet::cpp::Context(mxnet::cpp::Context::cpu()));
      holder.context_.reset(new mxnet::cpp::Context(mxnet::cpp::Context::gpu()));
      return Status::Ok();
    };
    return MxnetRunnerHolderManager::GetRunner(id, &holder, init);
  }

  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) {
    std::vector<Tensor> inputs;
    XDL_CHECK_STATUS(ctx->GetInputList("inputs", &inputs));
    XDL_CHECK_COND(inputs.size() == holder->var_names_.size(), 
                   Status::Internal("input size not equal"));
    MxnetRunner::InputList input_list;
    input_list.resize(holder->var_names_.size());
    for (size_t i = 0; i < holder->var_names_.size(); ++i) {
      input_list[i] = std::make_pair(holder->var_names_[i], inputs[i]);
    }

    std::vector<mxnet::cpp::NDArray> init_grads;
    if (holder->has_init_grad_) {
      Tensor init_grad;
      mxnet::cpp::NDArray converted_init_grad;
      XDL_CHECK_STATUS(ctx->GetInput("init_grad", &init_grad));      
      XDL_CHECK_STATUS(XDL2MX::ConvertTensor(
                           holder->context_.get(), 
                           init_grad, &converted_init_grad));
      XDL_CHECK_STATUS(XDL2MX::CopyGpuTensorSync(init_grad, &converted_init_grad));
      init_grads.clear();
      init_grads.push_back(converted_init_grad);
      holder->mxnet_runner_->SetInitGrad(&init_grads);
    }

    MxnetRunner::DataList outputs;
    MxnetRunner::DataList gradients;    
    XDL_CHECK_STATUS(holder->mxnet_runner_->Run(
                         input_list, 
                         &outputs, 
                         &gradients, 
                         stream->GetInternal()));
    std::vector<Tensor> output_list;
    output_list.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      MX2XDL::ConvertGpuTensorAsync(outputs[i], 
                                    &output_list[i], 
                                    stream->GetInternal());
    }

    std::vector<Tensor> grad_list;
    grad_list.resize(gradients.size());
    for (size_t i = 0; i < gradients.size(); ++i) {
      MX2XDL::ConvertGpuTensorAsync(gradients[i], 
                                    &grad_list[i],
                                    stream->GetInternal());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream->GetInternal()));    
    ctx->SetOutputList("targets", output_list);
    ctx->SetOutputList("gradients", grad_list);    
    return Status::Ok();
  }

 private:
  MxnetRunnerHolder* holder;
};

XDL_DEFINE_OP(MxnetBackendGpuOp)
  .InputListV2("inputs", "input_type")
  .Input("init_grad", DataType::kFloat)
  .OutputList("targets", DataType::kFloat, "target_size")
  .OutputList("gradients", DataType::kFloat, "gradient_size")
  .Attr("input_type", AttrValue::kDataTypeList)
  .Attr("target_size", AttrValue::kInt)
  .Attr("gradient_size", AttrValue::kInt)
  .Attr("var_name_str", AttrValue::kString)
  .Attr("device_type", AttrValue::kString)
  .Attr("is_training", AttrValue::kBool)
  .Attr("has_init_grad", AttrValue::kBool)
  .Attr("graph_def", AttrValue::kString)
  .Attr("id", AttrValue::kInt);

XDL_REGISTER_KERNEL(MxnetBackendOp, MxnetBackendGpuOp).Device("GPU");

} // namespace xdl
