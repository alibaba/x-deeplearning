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

#include <omp.h>

#include "xdl/core/lib/status.h"
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/backend/device_singleton.h"
#include "xdl/core/backend/mxnet/mxnet_runner.h"
#include "xdl/core/backend/mxnet/convert_utils.h"

namespace xdl {

class MxnetBackendOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) {
    std::string var_name;
    XDL_CHECK_STATUS(ctx->GetAttr("var_name_str", &var_name));
    XDL_CHECK_STATUS(ctx->GetAttr("graph_def", &graph_def_));
    XDL_CHECK_STATUS(ctx->GetAttr("device_type", &device_type_));
    XDL_CHECK_STATUS(ctx->GetAttr("gradient_size", &gradient_size_));
    XDL_CHECK_STATUS(ctx->GetAttr("is_training", &is_training_));
    XDL_CHECK_STATUS(ctx->GetAttr("has_init_grad", &has_init_grad_));
    XDL_CHECK_COND(device_type_ == "cpu", 
                   Status::ArgumentError("device type must be cpu"));
    var_names_ = StringUtils::split(var_name, ",");
    mxnet_runner_.reset(new MxnetRunner(is_training_));
    XDL_CHECK_STATUS(mxnet_runner_->Init(graph_def_, device_type_));
    context_.reset(new mxnet::cpp::Context(mxnet::cpp::Context::cpu()));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    std::vector<Tensor> inputs;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("inputs", &inputs), done);
    XDL_CHECK_COND_ASYNC(inputs.size() == var_names_.size(), 
                         Status::Internal("input size not equal"), 
                         done);
    MxnetRunner::InputList input_list;
    input_list.resize(var_names_.size());
    for (size_t i = 0; i < var_names_.size(); ++i) {
      input_list[i] = std::make_pair(var_names_[i], inputs[i]);
    }

    std::vector<mxnet::cpp::NDArray> init_grads;
    if (has_init_grad_) {
      Tensor init_grad;
      mxnet::cpp::NDArray converted_init_grad;
      XDL_CHECK_STATUS_ASYNC(ctx->GetInput("init_grad", &init_grad), done);      
      XDL_CHECK_STATUS_ASYNC(
          XDL2MX::ConvertTensor(
              context_.get(), 
              init_grad, &converted_init_grad), done);
      XDL_CHECK_STATUS_ASYNC(
          XDL2MX::CopyTensor(init_grad, &converted_init_grad), done);
      init_grads.clear();
      init_grads.push_back(converted_init_grad);
      mxnet_runner_->SetInitGrad(&init_grads);
    }

    MxnetRunner::DataList outputs;
    MxnetRunner::DataList gradients;    
    XDL_CHECK_STATUS_ASYNC(
        mxnet_runner_->Run(input_list, &outputs, &gradients), done);
    std::vector<Tensor> output_list;
    output_list.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      MX2XDL::ConvertTensor(outputs[i], &output_list[i]);
    }

    ctx->SetOutputList("targets", output_list);
    std::vector<Tensor> grad_list;
    grad_list.resize(gradients.size());
    //#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < gradients.size(); ++i) {
      MX2XDL::ConvertTensor(gradients[i], &grad_list[i]);
    }

    ctx->SetOutputList("gradients", grad_list);    
    done(Status::Ok());
  }

 private:
  std::unique_ptr<MxnetRunner> mxnet_runner_;
  std::string graph_def_;
  std::string device_type_;
  std::vector<std::string> var_names_;
  std::unique_ptr<mxnet::cpp::Context> context_;
  bool is_training_;
  bool has_init_grad_;
  int64_t gradient_size_;
};

XDL_DEFINE_OP(MxnetBackendOp)
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
  .Attr("graph_def", AttrValue::kString);

XDL_REGISTER_KERNEL(MxnetBackendOp, MxnetBackendOp).Device("CPU");

} // namespace xdl
