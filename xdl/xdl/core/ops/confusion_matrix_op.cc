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

#include "xdl/core/utils/logging.h"

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/backend/device_singleton.h"

namespace xdl {

template<typename T>
inline bool ToBool(T label) {
  return label;
}

template <typename T>
class ConfusionMatrixOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("num_thresholds", &num_thresholds_));
    thresholds_.reserve(num_thresholds_);
    InitThreholds();
    return Status::Ok();
  }

  void InitThreholds() {
    const float kEpsilon = 1e-7;
    thresholds_.push_back(-kEpsilon);
    for (size_t i = 0; i < num_thresholds_ - 2; ++i) {
      thresholds_.push_back((i + 1) * 1.0 / (num_thresholds_ - 1));
    }

    thresholds_.push_back(1.0 + kEpsilon);
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor predictions;
    XDL_CHECK_STATUS(ctx->GetInput(0, &predictions));    
    Tensor labels;
    XDL_CHECK_STATUS(ctx->GetInput(1, &labels));    
    std::vector<Tensor> matrix;
    ComputeInternal(predictions, labels, &matrix);
    ctx->SetOutputList("output", matrix);
    return Status::Ok();
  }

  void ComputeInternal(const Tensor& predictions, 
                       const Tensor& labels,
                       std::vector<Tensor>* result) {
    result->reserve(4);
    for (size_t i = 0; i < 4; ++i) {
      result->emplace_back();
      InitOutput(&(result->back()));
    }

    size_t pdim = predictions.Shape().Size();
    size_t ldim = labels.Shape().Size();
    XDL_CHECK(pdim == 1 || pdim == 2) << "predictions must be 1 or 2 dim";
    XDL_CHECK(pdim == ldim && predictions.Shape()[0] == labels.Shape()[0]) << 
      "predictions and labels must have same dim";
    if (pdim == 2) {
      XDL_CHECK(predictions.Shape()[1] == 1 && labels.Shape()[1] == 1) << 
        "second dim must be 1";
    }

    int64_t dim0 = predictions.Shape()[0];
    float* p_base = predictions.Raw<float>();
    T* l_base = labels.Raw<T>();
    int64_t* tp = result->at(0).Raw<int64_t>();
    int64_t* fp = result->at(1).Raw<int64_t>();
    int64_t* tn = result->at(2).Raw<int64_t>();
    int64_t* fn = result->at(3).Raw<int64_t>();
    for (int64_t i = 0; i < dim0; ++i) {
      float prediction = p_base[i];
      T label = l_base[i];
      for (size_t j = 0; j < thresholds_.size(); ++j) {
        if (prediction > thresholds_[j]) {
          if (ToBool(label)) tp[j]++; else fp[j]++;
        } else {
          if (ToBool(label)) fn[j]++; else tn[j]++;
        }
      }
    }
  }

  void InitOutput(Tensor* output) {
    *output = Tensor(DeviceSingleton::CpuInstance(), 
                     TensorShape({(size_t)num_thresholds_}),
                     DataType::kInt64);
    memset(output->Raw<void>(), 0, 
           output->Shape().NumElements() * SizeOfType(output->Type()));
  }

 private:
  int64_t num_thresholds_;
  std::vector<float> thresholds_;
};

template<>
inline bool ToBool(float label) {
  const float kEpsilon = 1e-6;
  return fabs(label - 1.0) < kEpsilon;
}

template<>
inline bool ToBool(double label) {
  const double kEpsilon = 1e-6;
  return fabs(label - 1.0) < kEpsilon;
}

XDL_DEFINE_OP(ConfusionMatrixOp)
  .Input("predictions", DataType::kFloat)
  .Input("labels", "dtype")
  .OutputList("output", DataType::kInt64, 4)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("num_thresholds", AttrValue::kInt);

XDL_REGISTER_KERNEL(ConfusionMatrixOp, ConfusionMatrixOp<int8_t>)
  .Device("CPU")
  .AttrDataType<int8_t>("dtype");

XDL_REGISTER_KERNEL(ConfusionMatrixOp, ConfusionMatrixOp<int16_t>)
  .Device("CPU")
  .AttrDataType<int16_t>("dtype");

XDL_REGISTER_KERNEL(ConfusionMatrixOp, ConfusionMatrixOp<int32_t>)
  .Device("CPU")
  .AttrDataType<int32_t>("dtype");

XDL_REGISTER_KERNEL(ConfusionMatrixOp, ConfusionMatrixOp<float>)
  .Device("CPU")
  .AttrDataType<float>("dtype");

XDL_REGISTER_KERNEL(ConfusionMatrixOp, ConfusionMatrixOp<double>)
  .Device("CPU")
  .AttrDataType<double>("dtype");
}  // namespace xdl
