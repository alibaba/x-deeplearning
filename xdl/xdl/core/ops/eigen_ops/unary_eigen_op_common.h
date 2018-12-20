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

#include "xdl/core/framework/eigen_op.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/op_define.h"

namespace xdl {

template <typename UnaryOp, typename Tin, typename Tout>
struct UnaryFunctor {
  template<typename Device>
  Status operator()(Device* device, OpKernelContext* ctx) {
    Tensor in_tensor, out_tensor;
    XDL_CHECK_STATUS(ctx->GetInput(0, &in_tensor));
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, in_tensor.Shape(), &out_tensor));
    size_t size = in_tensor.Shape().NumElements();
    Eigen::TensorMap<Eigen::Tensor<Tin, 1>> in(in_tensor.Raw<Tin>(), size);
    Eigen::TensorMap<Eigen::Tensor<Tout, 1>> out(out_tensor.Raw<Tout>(), size);
    out.device(*device) = in.unaryExpr(UnaryOp());
  }
};

}

#define XDL_REGISTER_UNARY_EIGEN_OP(OP, FUNCTOR, TYPE, OUT_TYPE)  \
    XDL_REGISTER_KERNEL(OP,                                       \
        ::xdl::EigenOp<::xdl::CpuDevice,                          \
        ::xdl::UnaryFunctor<                                      \
            FUNCTOR<::xdl::CpuDevice, TYPE, OUT_TYPE>,            \
            TYPE, OUT_TYPE>>)                                     \
      .Device("CPU")                                              \
      .AttrDataType<TYPE>("dtype");                               \
    XDL_REGISTER_KERNEL(OP,                                       \
        ::xdl::EigenOp<::xdl::GpuDevice,                          \
        ::xdl::UnaryFunctor<                                      \
            FUNCTOR<::xdl::GpuDevice, TYPE, OUT_TYPE>,            \
            TYPE, OUT_TYPE>>)                                     \
      .Device("GPU")                                              \
      .AttrDataType<TYPE>("dtype");

#define XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, TYPE) \
    XDL_REGISTER_UNARY_EIGEN_OP(OP, FUNCTOR, TYPE, TYPE)

#define XDL_REGISTER_CALC_UNARY_EIGEN_OP_SIMPLE(OP, FUNCTOR) \
    XDL_DEFINE_OP(OP)                                        \
      .Attr("dtype", ::xdl::AttrValue::kDataType)            \
      .Input("input", "dtype")                               \
      .Output("out", "dtype");                               \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, int64_t)   \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, int32_t)   \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, int16_t)   \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, int8_t)    \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, float)     \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, double)

#define XDL_REGISTER_FLOAT_CALC_UNARY_EIGEN_OP_SIMPLE(OP, FUNCTOR) \
    XDL_DEFINE_OP(OP)                                              \
      .Attr("dtype", ::xdl::AttrValue::kDataType)                  \
      .Input("input", "dtype")                                     \
      .Output("out", "dtype");                                     \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, float)           \
    XDL_REGISTER_CALC_UNARY_EIGEN_OP(OP, FUNCTOR, double)
