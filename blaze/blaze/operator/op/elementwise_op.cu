/*
 * \file elementwise_op.cu
 * \desc The elementwise operator on gpu implementation
 */
#include "blaze/operator/op/elementwise_op.h"
#include "blaze/math/elementwise/broadcast_elementwise.h" 
#include "blaze/math/elementwise/gpu_kernel_launcher.h"
#include "blaze/math/elementwise/elementwise_kernel.h"

namespace blaze {

#ifndef RUN_ELEMENTWISE_OP_KERNEL
#define RUN_ELEMENTWISE_OP_KERNEL(kernel)                                     \
  do {                                                                        \
    Blob* a = this->Input(0);                                                 \
    Blob* b = this->InputSize() < 2 ? this->Output(0) : this->Input(1);       \
    Blob* c = this->Output(0);                                                \
                                                                              \
    CheckValid();                                                             \
    Reshape();                                                                \
    bool need_broadcast = NeedBroadcast();                                    \
                                                                              \
    TYPE_SWITCH_ON_CUDA(a->data_type(), DType, {                              \
       ElementwiseParam<DType> params(a->as<DType>(), a->size(), a->shape(),  \
                                      b->as<DType>(), b->size(), b->shape(),  \
                                      c->as<DType>(), c->size(), c->shape()); \
       if (need_broadcast) {                                                  \
         Broadcast_##kernel(params, context());                               \
         continue;                                                            \
       }                                                                      \
       dim3 grid, block;                                                      \
       block.x = GetThreadsNum(c->size());                                    \
       grid.x = CUDA_GET_BLOCKS(c->size(), block.x);                          \
       cudaStream_t stream = this->context_.cuda_stream();                    \
       void* params_dptr = reinterpret_cast<void*>(&params);                  \
       CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&kernel<DType>),   \
                                   grid,                                      \
                                   block,                                     \
                                   reinterpret_cast<void**>(&params_dptr),    \
                                   0,                                         \
                                   stream));                                  \
    });                                                                       \
  } while (0)
#endif
 
template <typename DType, class Context>
void Broadcast_AddKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Sum, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void AddKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] + params.y[index % params.y_n];
  }
}

template <>
bool ElementwiseAddOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(AddKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Add, ElementwiseAddOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_SubKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Sub, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void SubKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] - params.y[index % params.y_n];
  }
}

template <>
bool ElementwiseSubOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(SubKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Sub, ElementwiseSubOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_MulKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Mul, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void MulKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] * params.y[index % params.y_n];
  }
}

template <>
bool ElementwiseMulOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MulKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Mul, ElementwiseMulOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_DivKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Div, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void DivKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] / params.y[index % params.y_n];
  }
}

template <>
bool ElementwiseDivOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(DivKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Div, ElementwiseDivOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_EqualKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Equal, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void EqualKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] == params.y[index % params.y_n] ? 1 : 0;
  }
}

template <>
bool ElementwiseEqualOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(EqualKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Equal, ElementwiseEqualOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_NotEqualKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::NotEqual, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void NotEqualKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] == params.y[index % params.y_n] ? 0 : 1;
  }
}

template <>
bool ElementwiseNotEqualOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(NotEqualKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(NotEqual, ElementwiseNotEqualOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_MaxKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Max, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void MaxKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] > params.y[index % params.y_n] ?
      params.x[index % params.x_n] : params.y[index % params.y_n];
  }
}

template <>
bool ElementwiseMaxOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MaxKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Max, ElementwiseMaxOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_MinKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Min, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void MinKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n] > params.y[index % params.y_n] ? 
      params.y[index % params.y_n] : params.x[index % params.x_n];
  }
}

template <>
bool ElementwiseMinOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MinKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(Min, ElementwiseMinOp<CUDAContext>);

template <typename DType, class Context>
void Broadcast_BroadcastToKernel(const ElementwiseParam<DType>& params,
                const Context& context) {
  bool res = broadcast::BroadcastCompute<DType, broadcast::Assign, GpuKernelLauncher, CUDAContext>(
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context); 
}

template <typename DType>
__global__ void BroadcastToKernel(ElementwiseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.x[index % params.x_n];
  }
}

template <>
bool BroadcastToOp<CUDAContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(BroadcastToKernel);
  return true;
}

REGISTER_CUDA_OPERATOR(BroadcastTo, BroadcastToOp<CUDAContext>);

template <typename IType, typename DType, class Context>
void Broadcast_WhereKernel(const TernaryElementwiseParam<IType, DType>& params,
                           const Context& context) {
  bool res = broadcast::BroadcastCompute<IType, DType, broadcast::Where, GpuKernelLauncher, CUDAContext>(
      params.condition, params.condition_shape,
      params.x, params.x_shape, params.y, params.y_shape,
      params.z, params.z_shape, context);
}

template <typename IType, typename DType>
__global__ void WhereKernel(TernaryElementwiseParam<IType, DType> params) {
  CUDA_KERNEL_LOOP(index, params.z_n) {
    params.z[index] = params.condition[index] > 0 ? params.x[index] : params.y[index];
  }
}

template <>
bool WhereOp<CUDAContext>::RunOnDevice() {
  Blob* condition = this->Input(0);
  Blob* x = this->Input(1);
  Blob* y = this->Input(2);
  Blob* z = this->Output(0);

  CheckValid();
  Reshape();
  bool need_broadcast = NeedBroadcast();

  ID_TYPE_SWITCH(condition->data_type(), IType, {
  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {
    TernaryElementwiseParam<IType, DType> params(condition->as<IType>(), condition->size(), condition->shape(),
                                                 x->as<DType>(), x->size(), x->shape(),
                                                 y->as<DType>(), y->size(), y->shape(),
                                                 z->as<DType>(), z->size(), z->shape());
    if (need_broadcast) {
      Broadcast_WhereKernel(params, context());
    } else {
      dim3 grid, block;
      block.x = GetThreadsNum(z->size());
      grid.x = GetBlockNum(CUDA_GET_BLOCKS(z->size(), block.x));
      cudaStream_t stream = this->context_.cuda_stream();
      void* params_dptr = reinterpret_cast<void*>(&params);
      CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&WhereKernel<IType, DType>),
                                  grid,
                                  block,
                                  reinterpret_cast<void**>(&params_dptr),
                                  0,
                                  stream));
    }
  });
  });

  return true;
}

REGISTER_CUDA_OPERATOR(Where, WhereOp<CUDAContext>);

#undef RUN_ELEMENTWISE_OP_KERNEL

}  // namespace blaze

