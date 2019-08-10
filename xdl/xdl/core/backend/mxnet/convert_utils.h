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

#ifndef XDL_BACKEND_MXNET_CONVERT_UTILS_H_
#define XDL_BACKEND_MXNET_CONVERT_UTILS_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <mxnet-cpp/MxNetCpp.h>

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/tensor.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "xdl/core/framework/gpu/gpu_stream.h"
#include "xdl/core/lib/common_defines.h"
#endif

namespace xdl {

class XDL2MX {
 public:
  static Status ConvertType(const xdl::DataType s, int* d);
  static Status ConvertShape(const xdl::TensorShape& s, 
                             std::vector<mx_uint>* d);
  static Status ConvertTensor(mxnet::cpp::Context* ctx, 
                              const xdl::Tensor& s, 
                              mxnet::cpp::NDArray* d);
  static Status ReshapeTensor(mxnet::cpp::Context* ctx,
                              const xdl::TensorShape& tshape, 
                              mxnet::cpp::NDArray* d);
  static Status CopyTensor(const xdl::Tensor& s,
                           mxnet::cpp::NDArray* d);
#ifdef USE_GPU
  static Status CopyGpuTensorAsync(const xdl::Tensor& s,
                                   mxnet::cpp::NDArray* d,
                                   cudaStream_t stream);

  static Status CopyGpuTensorSync(const xdl::Tensor& s,
                                  mxnet::cpp::NDArray* d);
#endif

  static int CompareShape(const xdl::TensorShape& s,
                          const std::vector<mx_uint>& d);
};

class MX2XDL {
 public:
  static Status ConvertType(const int s, xdl::DataType* d);
  static Status ConvertShape(const std::vector<mx_uint>& s, 
                             xdl::TensorShape* d);
  static Status ConvertTensor(mxnet::cpp::NDArray& s,
                              xdl::Tensor* d);
  static Status CopyTensor(mxnet::cpp::NDArray& s,
                           xdl::Tensor* d);

#ifdef USE_GPU
  static Status ConvertGpuTensorAsync(mxnet::cpp::NDArray& s,
                                      xdl::Tensor* d,
                                      cudaStream_t stream);

  static Status ConvertGpuTensorSync(mxnet::cpp::NDArray& s,
                                     xdl::Tensor* d);

  static Status CopyGpuTensorAsync(mxnet::cpp::NDArray& s, 
                                   xdl::Tensor* d,
                                   cudaStream_t stream);

  static Status CopyGpuTensorSync(mxnet::cpp::NDArray& s, 
                                  xdl::Tensor* d);
#endif
};

} // namespace xdl

#endif // XDL_BACKEND_MXNET_CONVERT_UTILS_H_
