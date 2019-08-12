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

#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/framework/slab_buddy_allocator.h"
#include "xdl/core/utils/logging.h"

#include <string>

namespace xdl {

void* GpuAllocator::Allocate(size_t size) {
  void* buf;
  std::cerr << "cuda alloc " << size << std::endl;
  cudaError_t e = cudaMalloc(&buf, size);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    XDL_LOG(ERROR) << "cudaMalloc failed, size=" << size
               << " avail=" << avail
               << " total=" << total
               << " err=" << e;
  }
  CudaStream::RunOrAbort(e, "Cuda Memory Allocate Error");
  return buf;
}

void GpuAllocator::Deallocate(void* buf) {
  CudaStream::RunOrAbort(cudaFree(buf), "Cuda Memory Deallocate Error");
}

std::string GpuDevice::DeviceType() {
  return "GPU";
}

Status GpuDevice::CreateDevice(const DeviceDef& def, Device** device) {
  auto iter = def.attr.find("id");
  if (iter == def.attr.end()) {
    *device = new GpuDevice(-1);
  } else {
    *device = new GpuDevice(atoi(iter->second.c_str()));
  }
  return Status::Ok();
}

GpuDevice::GpuDevice(int id)
    : Device(AllocatorManager::Instance()->Get(
              "GPU", []{
                Allocator* simple_allocator = new GpuAllocator;
                Allocator* slab_buddy_allocator = new SlabBuddyAllocator(
                    simple_allocator, 1ul << 30/*1G*/, 16ul << 10/*16K*/, 32
                  );
                simple_allocator->UnRef();
                return slab_buddy_allocator;
              })),
      stream_(CudaStreamManager::Instance()->GetCudaStream(id)) {}

void GpuOpKernel::Launch(OpKernelContext* ctx) {
#if 0
  Device* device = ctx->GetDevice();
  GpuDevice* gpu = dynamic_cast<GpuDevice*>(device);
  if (gpu == nullptr) {
    ctx->LaunchDone(Status::Internal("Run GpuOpKernel Without Gpu Device"));
  }
  CudaStream* stream = gpu->Stream();
  stream->Lock();
  Status st = LaunchKernel(ctx, stream);
  if (!st.IsOk()) {
    stream->Unlock();
    ctx->LaunchDone(st);
    ctx->RunDone(Status::Ok());
  } else {
    stream->AddCallback(ThreadPool::Global(),
        [=](Status st){ctx->RunDone(st);});
    stream->Unlock();
    ctx->LaunchDone(Status::Ok());
  }
#else
  CudaStream* stream = CudaStreams::GetInstance()->GetCudaStream();
  Status st = LaunchKernel(ctx, stream);
  if (!st.IsOk()) {
    ctx->LaunchDone(st);
    ctx->RunDone(Status::Ok());
  } else {
    stream->AddCallback(
        [=](Status st) { ctx->RunDone(st); ctx->LaunchDone(Status::Ok()); });
    //ctx->LaunchDone(Status::Ok());
  }
#endif
}

XDL_DEVICE_FACTORY(GPU, GpuDevice::CreateDevice);

}  // namespace xdl

