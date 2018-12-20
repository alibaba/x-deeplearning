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

template <typename BinaryOp, typename Tlhs, typename Trhs, typename Tout>
struct BroadcastBinaryFunctor {
  template <typename Device, int dim>
  Status InternalRun(Device* device,
      int64_t lhs_reshape[4], int64_t rhs_reshape[4], int64_t out_shape[4],
      int64_t lhs_broadcast[4], int64_t rhs_broadcast[4],
      bool lhs_need_bcast, bool rhs_need_bcast,
      Tlhs* lhs_raw, Trhs* rhs_raw, Tout* out_raw) {
    Eigen::array<Eigen::DenseIndex, dim> lhs_dim, rhs_dim, out_dim;
    for (int i = 0; i < dim; i++) {
      lhs_dim[i] = lhs_reshape[i];
    }
    for (int i = 0; i < dim; i++) {
      rhs_dim[i] = rhs_reshape[i];
    }
    for (int i = 0; i < dim; i++) {
      out_dim[i] = out_shape[i];
    }
    Eigen::TensorMap<Eigen::Tensor<Tlhs, dim>> lhs(lhs_raw, lhs_dim);
    Eigen::TensorMap<Eigen::Tensor<Trhs, dim>> rhs(rhs_raw, rhs_dim);
    Eigen::TensorMap<Eigen::Tensor<Tout, dim>> out(out_raw, out_dim);
    if (lhs_need_bcast && rhs_need_bcast) {
      Eigen::array<Eigen::DenseIndex, dim> lhs_bcast, rhs_bcast;
      for (int i = 0; i < dim; i++) {
        lhs_bcast[i]= lhs_broadcast[i];
      }
      for (int i = 0; i < dim; i++) {
        rhs_bcast[i]= rhs_broadcast[i];
      }
      out.device(*device) = lhs.broadcast(lhs_bcast).binaryExpr(
                            rhs.broadcast(rhs_bcast), BinaryOp());
    } else if (lhs_need_bcast && !rhs_need_bcast) {
      Eigen::array<Eigen::DenseIndex, dim> lhs_bcast;
      for (int i = 0; i < dim; i++) {
        lhs_bcast[i]= lhs_broadcast[i];
      }
      out.device(*device) = lhs.broadcast(lhs_bcast).binaryExpr(
                            rhs, BinaryOp());
    } else if (!lhs_need_bcast && rhs_need_bcast) {
      Eigen::array<Eigen::DenseIndex, dim> rhs_bcast;
      for (int i = 0; i < dim; i++) {
        rhs_bcast[i]= rhs_broadcast[i];
      }
      out.device(*device) = lhs.binaryExpr(
                            rhs.broadcast(rhs_bcast), BinaryOp());
    } else {
      out.device(*device) = lhs.binaryExpr(
                            rhs, BinaryOp());
    }
    return Status::Ok();
  }
  template <typename Device>
  Status InternalRun0(Device* device,
      int64_t lhs_reshape[4], int64_t rhs_reshape[4], int64_t out_shape[4],
      int64_t lhs_broadcast[4], int64_t rhs_broadcast[4],
      bool lhs_need_bcast, bool rhs_need_bcast,
      Tlhs* lhs_raw, Trhs* rhs_raw, Tout* out_raw) {
    Eigen::array<Eigen::DenseIndex, 0> lhs_dim, rhs_dim, out_dim;
    Eigen::TensorMap<Eigen::Tensor<Tlhs, 0>> lhs(lhs_raw, lhs_dim);
    Eigen::TensorMap<Eigen::Tensor<Trhs, 0>> rhs(rhs_raw, rhs_dim);
    Eigen::TensorMap<Eigen::Tensor<Tout, 0>> out(out_raw, out_dim);
    out.device(*device) = lhs.binaryExpr(
                          rhs, BinaryOp());
    return Status::Ok();
  }
  template<typename Device>
  Status operator()(Device* device, OpKernelContext* ctx) {
    Tensor lhs, rhs, out;
    XDL_CHECK_STATUS(ctx->GetInput(0, &lhs));
    XDL_CHECK_STATUS(ctx->GetInput(1, &rhs));
    auto lhs_shape = lhs.Shape().Dims();
    auto rhs_shape = rhs.Shape().Dims();
    bool lhs_need_bcast = false, rhs_need_bcast = false;
    int64_t lhs_reshape[4], rhs_reshape[4], out_shape[4],
           lhs_broadcast[4], rhs_broadcast[4];
    memcpy(lhs_reshape, &lhs_shape[0], lhs_shape.size() * sizeof(int64_t));
    memcpy(rhs_reshape, &rhs_shape[0], rhs_shape.size() * sizeof(int64_t));
    size_t dims = std::max(lhs_shape.size(), rhs_shape.size());
    if (dims > 4) {
      return Status::Internal("Dim more than 4 is not supported");
    }
    for (size_t i = lhs_shape.size(); i < dims; i++) {
      lhs_reshape[i] = 1;
    }
    for (size_t i = rhs_shape.size(); i < dims; i++) {
      rhs_reshape[i] = 1;
    }
    for (size_t i = 0; i < dims; i++) {
      lhs_broadcast[i] = lhs_reshape[i] != 1 ? 1 : rhs_reshape[i];
      lhs_need_bcast |= lhs_broadcast[i] != 1;
      rhs_broadcast[i] = rhs_reshape[i] != 1 ? 1 : lhs_reshape[i];
      rhs_need_bcast |= rhs_broadcast[i] != 1;
      out_shape[i] = lhs_reshape[i] != 1 ? lhs_reshape[i] : rhs_reshape[i];
      if (lhs_reshape[i] != 1 && rhs_reshape[i] != 1
          && lhs_reshape[i] != rhs_reshape[i]) {
        return Status::ArgumentError(
            "Dim Error " + lhs.Shape().DebugString() +
            " vs " + rhs.Shape().DebugString());
      }
    }
    XDL_CHECK_STATUS(ctx->AllocateOutput(0,
          TensorShape(std::vector<size_t>(out_shape, out_shape + dims)),
          &out));
    Tlhs* lhs_raw = lhs.Raw<Tlhs>();
    Trhs* rhs_raw = rhs.Raw<Trhs>();
    Tout* out_raw = out.Raw<Tout>();
    switch (dims) {
      case 0:
        XDL_CHECK_STATUS(XDL_SINGLE_ARG(
              InternalRun0(
                  device, lhs_reshape, rhs_reshape, out_shape,
                  lhs_broadcast, rhs_broadcast, lhs_need_bcast, rhs_need_bcast,
                  lhs_raw, rhs_raw, out_raw)));
        return Status::Ok();
      case 1:
        XDL_CHECK_STATUS(XDL_SINGLE_ARG(
              InternalRun<Device, 1>(
                  device, lhs_reshape, rhs_reshape, out_shape,
                  lhs_broadcast, rhs_broadcast, lhs_need_bcast, rhs_need_bcast,
                  lhs_raw, rhs_raw, out_raw)));
        return Status::Ok();
      case 2:
        XDL_CHECK_STATUS(XDL_SINGLE_ARG(
              InternalRun<Device, 2>(
                  device, lhs_reshape, rhs_reshape, out_shape,
                  lhs_broadcast, rhs_broadcast, lhs_need_bcast, rhs_need_bcast,
                  lhs_raw, rhs_raw, out_raw)));
        return Status::Ok();
      case 3:
        XDL_CHECK_STATUS(XDL_SINGLE_ARG(
              InternalRun<Device, 3>(
                  device, lhs_reshape, rhs_reshape, out_shape,
                  lhs_broadcast, rhs_broadcast, lhs_need_bcast, rhs_need_bcast,
                  lhs_raw, rhs_raw, out_raw)));
        return Status::Ok();
      case 4:
        XDL_CHECK_STATUS(XDL_SINGLE_ARG(
              InternalRun<Device, 4>(
                  device, lhs_reshape, rhs_reshape, out_shape,
                  lhs_broadcast, rhs_broadcast, lhs_need_bcast, rhs_need_bcast,
                  lhs_raw, rhs_raw, out_raw)));
        return Status::Ok();
      default:
        return Status::Internal("reach logical unreachable code");
    }
  }
};

}

#define XDL_REGISTER_BINARY_EIGEN_OP(OP, FUNCTOR, TYPE, RST_TYPE)  \
    XDL_REGISTER_KERNEL(OP,                                        \
        ::xdl::EigenOp<::xdl::CpuDevice,                           \
        ::xdl::BroadcastBinaryFunctor<                             \
            FUNCTOR<::xdl::CpuDevice, TYPE, TYPE, RST_TYPE>,       \
            TYPE, TYPE, RST_TYPE>>)                                \
      .Device("CPU")                                               \
      .AttrDataType<TYPE>("dtype");                                \
    XDL_REGISTER_KERNEL(OP,                                        \
        ::xdl::EigenOp<::xdl::GpuDevice,                           \
        ::xdl::BroadcastBinaryFunctor<                             \
            FUNCTOR<::xdl::CpuDevice, TYPE, TYPE, RST_TYPE>,       \
            TYPE, TYPE, RST_TYPE>>)                                \
      .Device("GPU")                                               \
      .AttrDataType<TYPE>("dtype");

#define XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, TYPE) \
    XDL_REGISTER_BINARY_EIGEN_OP(OP, FUNCTOR, TYPE, TYPE)

#define XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, TYPE) \
    XDL_REGISTER_BINARY_EIGEN_OP(OP, FUNCTOR, TYPE, bool)

#define XDL_REGISTER_INT_CALC_BINARY_EIGEN_OP_SIMPLE(OP, FUNCTOR) \
    XDL_DEFINE_OP(OP)                                             \
      .Attr("dtype", ::xdl::AttrValue::kDataType)                 \
      .Input("lhs", "dtype")                                      \
      .Input("rhs", "dtype")                                      \
      .Output("out", "dtype");                                    \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int64_t)       \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int32_t)       \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int16_t)       \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int8_t)

#define XDL_REGISTER_CALC_BINARY_EIGEN_OP_SIMPLE(OP, FUNCTOR) \
    XDL_DEFINE_OP(OP)                                         \
      .Attr("dtype", ::xdl::AttrValue::kDataType)             \
      .Input("lhs", "dtype")                                  \
      .Input("rhs", "dtype")                                  \
      .Output("out", "dtype");                                \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int64_t)   \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int32_t)   \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int16_t)   \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, int8_t)    \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, float)     \
    XDL_REGISTER_CALC_BINARY_EIGEN_OP(OP, FUNCTOR, double)

#define XDL_REGISTER_BOOL_BINARY_EIGEN_OP_SIMPLE(OP, FUNCTOR) \
    XDL_DEFINE_OP(OP)                                         \
      .Attr("dtype", ::xdl::AttrValue::kDataType)             \
      .Input("lhs", "dtype")                                  \
      .Input("rhs", "dtype")                                  \
      .Output("out", ::xdl::DataType::kBool);                 \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, int64_t)   \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, int32_t)   \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, int16_t)   \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, int8_t)    \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, float)     \
    XDL_REGISTER_BOOL_BINARY_EIGEN_OP(OP, FUNCTOR, double)
