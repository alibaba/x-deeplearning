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

#include "xdl/core/ops/add_sparse_gradient.h"

#include "xdl/core/backend/device_singleton.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/framework/gpu/gpu_device.h"

namespace xdl {

template <typename T, typename I>
class SparseGradAddGpuOp : public GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override {
    std::vector<Tensor> in_grads, in_ids;
    XDL_CHECK_STATUS(ctx->GetInputList("in_grads", &in_grads));
    XDL_CHECK_STATUS(ctx->GetInputList("in_ids", &in_ids));
    XDL_CHECK_COND(in_grads.size() == in_ids.size(),
                   Status::ArgumentError("grads and ids size not equal"));
    for (size_t i = 0; i < in_grads.size(); ++i) {
      XDL_CHECK_COND(in_grads[i].Shape().Size() == 2,
                     Status::ArgumentError("grad input dim must be 2"));
      XDL_CHECK_COND(in_ids[i].Shape().Size() == 1 ||
                     in_ids[i].Shape().Size() == 2,
                     Status::ArgumentError("id input dim must be 1 or 2"));
      XDL_CHECK_COND(in_grads[i].Shape()[0] == in_ids[i].Shape()[0],
                     Status::ArgumentError("grad dim 0 not equal to id dim 0"));
    }
    if (in_grads.empty()) return Status::Ok();

    // copy data to host
    std::vector<Tensor> host_grads, host_ids;
    for (size_t i = 0; i < in_ids.size(); ++i) {
      Tensor grad(DeviceSingleton::CpuInstance(), in_grads[i].Shape(),
                  in_grads[i].Type());
      CUDA_CHECK(cudaMemcpyAsync(grad.Raw<T>(), in_grads[i].Raw<T>(),
                                 sizeof(T) * in_grads[i].Shape().NumElements(),
                                 cudaMemcpyDeviceToHost,
                                 stream->GetInternal()));
      host_grads.push_back(grad);
      Tensor id(DeviceSingleton::CpuInstance(), in_ids[i].Shape(),
                in_ids[i].Type());
      CUDA_CHECK(cudaMemcpyAsync(id.Raw<I>(), in_ids[i].Raw<I>(),
                                 sizeof(I) * in_ids[i].Shape().NumElements(),
                                 cudaMemcpyDeviceToHost,
                                 stream->GetInternal()));
      host_ids.push_back(id);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream->GetInternal()));
    // add sparse on host
    Tensor host_out_grad, host_out_id;
    HostAddSparse<T, I>(host_grads, host_ids, &host_out_grad, &host_out_id);
    // copy host data to device
    Tensor out_grad, out_id;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, host_out_grad.Shape(), &out_grad));
    XDL_CHECK_STATUS(ctx->AllocateOutput(1, host_out_id.Shape(), &out_id));
    CUDA_CHECK(cudaMemcpyAsync(out_grad.Raw<T>(),
                               host_out_grad.Raw<T>(),
                               sizeof(T) * host_out_grad.Shape().NumElements(),
                               cudaMemcpyHostToDevice,
                               stream->GetInternal()));
    CUDA_CHECK(cudaMemcpyAsync(out_id.Raw<I>(),
                               host_out_id.Raw<I>(),
                               sizeof(I) * host_out_id.Shape().NumElements(),
                               cudaMemcpyHostToDevice,
                               stream->GetInternal()));

    return Status::Ok();
  }
};

#define REGISTER_GPU_KERNEL(T, I)                                \
  XDL_REGISTER_KERNEL(SparseGradAddOp, SparseGradAddGpuOp<T, I>) \
  .Device("GPU")                                                 \
  .AttrDataType<T>("dtype")                                      \
  .AttrDataType<I>("itype");

REGISTER_GPU_KERNEL(float, int32_t);
REGISTER_GPU_KERNEL(float, int64_t);
REGISTER_GPU_KERNEL(double, int32_t);
REGISTER_GPU_KERNEL(double, int64_t);

#undef REGISTER_GPU_KERNEL

}  // namespace xdl
