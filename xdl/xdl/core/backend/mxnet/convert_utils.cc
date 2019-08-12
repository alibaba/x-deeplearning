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

#include "xdl/core/backend/mxnet/convert_utils.h"
#include "xdl/core/backend/device_singleton.h"

namespace xdl {

Status XDL2MX::ConvertType(const xdl::DataType s, int* d) {
  switch(s) {
  case DataType::kInt8: 
    *d = 5; 
    return Status::Ok();
  case DataType::kInt32: 
    *d = 4; 
    return Status::Ok();
  case DataType::kInt64: 
    *d = 6; 
    return Status::Ok();
  case DataType::kFloat: 
    *d = 0; 
    return Status::Ok();
  case DataType::kDouble: 
    *d = 1; 
    return Status::Ok();
  default:
    return Status::Internal("unsupported type:" + std::to_string(s));
  }
}

Status XDL2MX::ConvertShape(const xdl::TensorShape& s, 
                            std::vector<mx_uint>* d) {
  for (auto dim: s.Dims()) {
    d->push_back(static_cast<mx_uint>(dim));
  }

  return Status::Ok();
}

Status XDL2MX::ReshapeTensor(mxnet::cpp::Context* ctx,
                             const xdl::TensorShape& tshape, 
                             mxnet::cpp::NDArray* d) {
  int type;
  std::vector<mx_uint> shape;
  XDL_CHECK_STATUS(ConvertShape(tshape, &shape));

  *d = d->Reshape(mxnet::cpp::Shape(shape));

  return Status::Ok();
}

Status XDL2MX::ConvertTensor(mxnet::cpp::Context* ctx,
                             const xdl::Tensor& s, 
                             mxnet::cpp::NDArray* d) {
  int type;
  std::vector<mx_uint> shape;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  NDArrayHandle handle;
  XDL_CHECK_COND(0 == MXNDArrayCreateEx(shape.data(), 
                                        shape.size(), 
                                        ctx->GetDeviceType(),
                                        ctx->GetDeviceId(),
                                        false,
                                        type,
                                        &handle), 
                 Status::Internal("create NDArray failed!"));
  *d = mxnet::cpp::NDArray(handle);
  return Status::Ok();
}

Status XDL2MX::CopyTensor(const xdl::Tensor& s,
                          mxnet::cpp::NDArray* d) {
  d->SyncCopyFromCPU(s.Raw<mx_float>(), s.Shape().NumElements());
  return Status::Ok();
}

#ifdef USE_GPU
Status XDL2MX::CopyGpuTensorAsync(const xdl::Tensor& s,
                                  mxnet::cpp::NDArray* d,
                                  cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>((const void*)d->GetData()),
                             s.Raw<void>(),
                             s.Shape().NumElements() * SizeOfType(s.Type()),
                             cudaMemcpyDeviceToDevice,
                             stream));
  return Status::Ok();
}

Status XDL2MX::CopyGpuTensorSync(const xdl::Tensor& s,
                                 mxnet::cpp::NDArray* d) {
  CUDA_CHECK(cudaMemcpy(const_cast<void*>((const void*)d->GetData()),
                        s.Raw<void>(),
                        s.Shape().NumElements() * SizeOfType(s.Type()),
                        cudaMemcpyDeviceToDevice));
  return Status::Ok();
}
#endif

int XDL2MX::CompareShape(const xdl::TensorShape& s,
                         const std::vector<mx_uint>& d) {
  const std::vector<size_t>& dims = s.Dims();
  const size_t size = dims.size();
  if (size != d.size()) {
    return (int)size - (int)d.size();
  }
  for (size_t i = 0; i < size; ++i) {
    if ((mx_uint)dims[i] != d[i]) {
      return (int)dims[i] - (int)d[i];
    }
  }
  return 0;
}

Status MX2XDL::ConvertType(const int s, xdl::DataType* d) {
  switch(s) {
  case 5: 
    *d = DataType::kInt8; 
    return Status::Ok();
  case 4: 
    *d = DataType::kInt32; 
    return Status::Ok();
  case 6: 
    *d = DataType::kInt64; 
    return Status::Ok();
  case 0: 
    *d = DataType::kFloat; 
    return Status::Ok();
  case 1: 
    *d = DataType::kDouble; 
    return Status::Ok();
  default:
    return Status::Internal("unsupported type:" + std::to_string(s));
  }
}

Status MX2XDL::ConvertShape(const std::vector<mx_uint>& s, xdl::TensorShape* d) {
  std::vector<size_t> dims;
  for (auto dim: s) {
    dims.push_back(static_cast<size_t>(dim));
  }

  *d = TensorShape(dims);
  return Status::Ok();
}

Status MX2XDL::ConvertTensor(mxnet::cpp::NDArray& s, xdl::Tensor* d) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.GetDType(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.GetShape(), &shape));
  *d = Tensor(DeviceSingleton::CpuInstance(), shape, type);
  CopyTensor(s, d);
  return Status::Ok();
}

Status MX2XDL::CopyTensor(mxnet::cpp::NDArray& s, xdl::Tensor* d) {
  s.SyncCopyToCPU(d->Raw<mx_float>(), d->Shape().NumElements());
  return Status::Ok();
}

#ifdef USE_GPU
Status MX2XDL::ConvertGpuTensorAsync(mxnet::cpp::NDArray& s, 
                                     xdl::Tensor* d,
                                     cudaStream_t stream) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.GetDType(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.GetShape(), &shape));
  *d = Tensor(DeviceSingleton::GpuInstance(), shape, type);
  CopyGpuTensorAsync(s, d, stream);
  return Status::Ok();
}

Status MX2XDL::CopyGpuTensorAsync(mxnet::cpp::NDArray& s, 
                                  xdl::Tensor* d,
                                  cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(d->Raw<void>(),
                             const_cast<void*>((const void*)s.GetData()),
                             d->Shape().NumElements() * SizeOfType(d->Type()),
                             cudaMemcpyDeviceToDevice,
                             stream));
  return Status::Ok();
}

Status MX2XDL::ConvertGpuTensorSync(mxnet::cpp::NDArray& s, 
                                    xdl::Tensor* d) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.GetDType(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.GetShape(), &shape));
  *d = Tensor(DeviceSingleton::GpuInstance(), shape, type);
  CopyGpuTensorSync(s, d);
  return Status::Ok();
}

Status MX2XDL::CopyGpuTensorSync(mxnet::cpp::NDArray& s, 
                                 xdl::Tensor* d) {
  CUDA_CHECK(cudaMemcpy(d->Raw<void>(),
                        const_cast<void*>((const void*)s.GetData()),
                        d->Shape().NumElements() * SizeOfType(d->Type()),
                        cudaMemcpyDeviceToDevice));
  return Status::Ok();
}

#endif

} // namespace xdl

