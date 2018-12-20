/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/utils/logging.h"

namespace xdl {

template <typename T>
class ConstantGpuOp : public GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    TensorShape shape;
    std::string value;
    XDL_CHECK_STATUS(ctx->GetAttr("shape", &shape));
    XDL_CHECK_STATUS(ctx->GetAttr("value", &value));
    XDL_CHECK_COND(shape.NumElements() * sizeof(T) == value.size(),
                   Status::ArgumentError("shape is not equal to value"));
    tensor_ = Tensor(ctx->GetDevice(), shape, DataTypeToEnum<T>::v());
    GpuDevice* device = dynamic_cast<GpuDevice*>(ctx->GetDevice());
    XDL_CHECK(device != nullptr) << "gpu devivce is nullptr";
    CUDA_CHECK(cudaMemcpyAsync(tensor_.Raw<T>(),
                               value.data(),
                               value.size(),
                               cudaMemcpyHostToDevice,
                               device->Stream()->GetInternal()));
    return Status::Ok();
  }

  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override {
    XDL_CHECK_STATUS(ctx->SetOutput(0, tensor_));
    return Status::Ok();
  }
 private:
  Tensor tensor_;
};

#define REGISTER_GPU_KERNEL(T)                     \
  XDL_REGISTER_KERNEL(_Constant, ConstantGpuOp<T>) \
  .Device("GPU")                                   \
  .AttrDataType<T>("dtype");

REGISTER_GPU_KERNEL(bool)
REGISTER_GPU_KERNEL(int8_t)
REGISTER_GPU_KERNEL(int16_t)
REGISTER_GPU_KERNEL(int32_t)
REGISTER_GPU_KERNEL(int64_t)
REGISTER_GPU_KERNEL(float)
REGISTER_GPU_KERNEL(double)

#undef REGISTER_GPU_KERNEL

}  // namespace
