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

#ifndef XDL_CORE_OPS_PS_OPS_CONVERT_UTILS_H_
#define XDL_CORE_OPS_PS_OPS_CONVERT_UTILS_H_

#include "ps-plus/common/tensor.h"
#include "ps-plus/common/status.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/framework/op_kernel.h"

namespace xdl {

class PS2XDL {
 public:
  static Status ConvertType(ps::DataType src, xdl::DataType* dst);
  static Status ConvertShape(const ps::TensorShape& src, xdl::TensorShape* dst);
  static Status ConvertTensor(const ps::Tensor& src, xdl::Tensor* dst);
  static Status ConvertTensorZC(ps::Tensor& src, xdl::Tensor* dst);  
  static Status ConvertStatus(const ps::Status& st);
  static Status ConvertTensorWithCtx(const ps::Tensor& src, 
                                     OpKernelContext* ctx, 
                                     int output_id);
};

class XDL2PS {
 public:
  static Status ConvertType(xdl::DataType src, ps::DataType* dst);
  static Status ConvertShape(const xdl::TensorShape& src, ps::TensorShape* dst);
  static Status ConvertTensor(const xdl::Tensor& src, ps::Tensor* dst);
  static Status ConvertTensorZC(xdl::Tensor& src, ps::Tensor* dst);  
  static ps::Status ConvertStatus(const Status& st);
};

} // namespace xdl

#endif // XDL_CORE_OPS_PS_OPS_CONVERT_UTILS_H_
