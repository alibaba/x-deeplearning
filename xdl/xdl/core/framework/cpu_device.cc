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

#include "xdl/core/framework/cpu_device.h"

#include <string>
#include <malloc.h>

namespace xdl {

void* CpuAllocator::Allocate(size_t size) {
  //return static_cast<void*>(new char[size]);
  return static_cast<void*>(memalign(64, ((size+63)/64)*64));
}

void CpuAllocator::Deallocate(void* buf) {
  //delete [] static_cast<char*>(buf);
  free(buf);
  return;
}

CpuDevice::CpuDevice()
    : Device(AllocatorManager::Instance()->Get(
             "CPU", [] ()->Allocator* { return new CpuAllocator; })) {}

std::string CpuDevice::DeviceType() {
  return "CPU";
}

Status CpuDevice::CreateDevice(const DeviceDef& def, Device** device) {
  *device = new CpuDevice;
  return Status::Ok();
}

XDL_DEVICE_FACTORY(CPU, CpuDevice::CreateDevice);

}  // namespace xdl

