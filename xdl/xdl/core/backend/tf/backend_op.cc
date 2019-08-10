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

#include <omp.h>

#include "xdl/core/lib/status.h"
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/backend/tf/tf_runner.h"
#include "xdl/core/backend/tf/convert_utils.h"

namespace xdl {

class TFBackendOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) {
    std::string input_op_name;
    std::string target_op_name;
    std::string gradient_op_name;
    std::string local_init_op_name;
    float gpu_memory_fraction;
    XDL_CHECK_STATUS(ctx->GetAttr("input_op_names", &input_op_name));
    XDL_CHECK_STATUS(ctx->GetAttr("target_op_names", &target_op_name));
    XDL_CHECK_STATUS(ctx->GetAttr("gradient_op_names", &gradient_op_name));
    XDL_CHECK_STATUS(ctx->GetAttr("local_init_op_names", &local_init_op_name));
    XDL_CHECK_STATUS(ctx->GetAttr("graph_def", &graph_def_));
    XDL_CHECK_STATUS(ctx->GetAttr("gpu_memory_fraction", &gpu_memory_fraction));    
    input_op_names_ = StringUtils::split(input_op_name, ",");
    target_op_names_ = StringUtils::split(target_op_name, ",");
    gradient_op_names_ = StringUtils::split(gradient_op_name, ",");
    local_init_op_names_ = StringUtils::split(local_init_op_name, ",");
    XDL_CHECK_STATUS(tf_runner_.Init(graph_def_, gpu_memory_fraction));
    XDL_CHECK_STATUS(InitLocalVariables());
    return Status::Ok();
  }

  Status InitLocalVariables() {
    if (!local_init_op_names_.empty()) {
      TFRunner::InputList tf_inputs;      
      std::vector<tensorflow::Tensor> results;
      XDL_CHECK_STATUS(tf_runner_.Run(tf_inputs, local_init_op_names_, &results));
    }

    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    using TensorList = std::vector<Tensor>;
    TensorList inputs;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("inputs", &inputs), done);
    TFRunner::InputList tf_inputs;
    size_t input_op_size = input_op_names_.size();
    tf_inputs.resize(input_op_size);
    #pragma omp parallel for
    for (size_t i = 0; i < input_op_size; ++i) {
      tensorflow::Tensor t;
      XDL2TF::ConvertTensor(inputs[i], &t);
      tf_inputs[i] = std::make_pair(input_op_names_[i], t);
    }

    size_t target_op_size = target_op_names_.size();
    size_t gradient_op_size = gradient_op_names_.size();
    std::vector<std::string> output_op_names;
    output_op_names.reserve(target_op_size + gradient_op_size);
    output_op_names.insert(output_op_names.end(), 
                           target_op_names_.begin(), 
                           target_op_names_.end());
    output_op_names.insert(output_op_names.end(), 
                           gradient_op_names_.begin(), 
                           gradient_op_names_.end());
    
	results_.clear();
    XDL_CHECK_STATUS_ASYNC(
        tf_runner_.Run(tf_inputs, output_op_names, &results_), 
        done);
    
    TensorList outputs;
    for (size_t i = 0; i < target_op_size; ++i) {
      Tensor t;
      XDL_CHECK_STATUS_ASYNC(TF2XDL::ConvertTensor(results_[i], &t), done);
      outputs.emplace_back(t);
    }

    ctx->SetOutputList("targets", outputs);
    TensorList gradients;
    gradients.resize(gradient_op_size);    
    #pragma omp parallel for
    for (size_t i = 0; i < gradient_op_size; ++i) {
      Tensor t;
      TF2XDL::ConvertTensor(results_[i + target_op_size], &t);
      gradients[i] = std::move(t);
    }

    ctx->SetOutputList("gradients", gradients);
    done(Status::Ok());
  }

 private:
  TFRunner tf_runner_;
  std::string graph_def_;
  std::vector<std::string> input_op_names_;
  std::vector<std::string> target_op_names_;
  std::vector<std::string> gradient_op_names_;
  std::vector<std::string> local_init_op_names_;
  std::vector<tensorflow::Tensor> results_;
};

XDL_DEFINE_OP(TFBackendOp)
  .InputListV2("inputs", "input_type")
  .OutputListV2("targets", "output_type")
  .OutputList("gradients", DataType::kFloat, "gradient_size")
  .Attr("input_type", AttrValue::kDataTypeList)
  .Attr("output_type", AttrValue::kDataTypeList)
  .Attr("input_op_names", AttrValue::kString)
  .Attr("target_op_names", AttrValue::kString)
  .Attr("gradient_op_names", AttrValue::kString)
  .Attr("local_init_op_names", AttrValue::kString)
  .Attr("gradient_size", AttrValue::kInt)
  .Attr("graph_def", AttrValue::kString)
  .Attr("gpu_memory_fraction", AttrValue::kFloat);

XDL_REGISTER_KERNEL(TFBackendOp, TFBackendOp).Device("CPU");

} // namespace xdl
