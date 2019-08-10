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
class IdentityGpuOp : public GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override {
    Tensor input;    
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, input.Shape(), &output));
    CUDA_CHECK(cudaMemcpy(output.Raw<char>(), input.Raw<char>(), 
                          output.Shape().NumElements() * SizeOfType(output.Type()),
                          cudaMemcpyDeviceToDevice));
    return Status::Ok();
  }
};

#define REGISTER_GPU_KERNEL(T)                     \
  XDL_REGISTER_KERNEL(IdentityOp, IdentityGpuOp<T>) \
  .Device("GPU")                                   \
  .AttrDataType<T>("dtype");

REGISTER_GPU_KERNEL(int8_t)
REGISTER_GPU_KERNEL(int16_t)
REGISTER_GPU_KERNEL(int32_t)
REGISTER_GPU_KERNEL(int64_t)
REGISTER_GPU_KERNEL(float)
REGISTER_GPU_KERNEL(double)

#undef REGISTER_GPU_KERNEL

}  // namespace
