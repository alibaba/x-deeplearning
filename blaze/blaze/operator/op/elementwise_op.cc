/*
 * \file elementwise_op.cc
 * \desc The elementwise operator on cpu implementation
 */
#include "blaze/operator/op/elementwise_op.h"
#include "blaze/operator/common_helper.h"
#include "blaze/math/elementwise/broadcast_elementwise.h" 
#include "blaze/math/elementwise/cpu_kernel_launcher.h"
#include "blaze/math/elementwise/elementwise_kernel.h"
#include "blaze/math/vml.h"

namespace blaze {

#ifndef RUN_ELEMENTWISE_OP_KERNEL
#define RUN_ELEMENTWISE_OP_KERNEL(kernel)                         \
  do {                                                            \
    Blob* a = this->Input(0);                                     \
    Blob* b = this->InputSize() < 2 ? this->Output(0) : this->Input(1);  \
    Blob* c = this->Output(0);                                    \
                                                                  \
    CheckValid();                                                 \
    Reshape();                                                    \
    bool need_broadcast = NeedBroadcast();                        \
                                                                  \
    TYPE_SWITCH(a->data_type(), DType, {                          \
      ElementwiseParam<DType> params(a->as<DType>(), a->size(), a->shape(),   \
                                     b->as<DType>(), b->size(), b->shape(),   \
                                     c->as<DType>(), c->size(), c->shape());  \
      return kernel(params, need_broadcast, context());           \
    });                                                           \
  } while (0)
#endif

template <typename DType, class Context>
bool AddKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Sum, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }

  VML_Add<DType, CPUContext>(params.z_n, params.x, params.y, params.z, nullptr); 
  return true;
}

template <>
bool  ElementwiseAddOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(AddKernel);
}

REGISTER_CPU_OPERATOR(Add, ElementwiseAddOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Add)
  .NumInputs(2)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
a + b = c
  )DOC");

template <typename DType, class Context>
bool SubKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Sub, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }

  VML_Sub<DType, CPUContext>(params.z_n, params.x, params.y, params.z, nullptr); 
  return true;
}

template <>
bool ElementwiseSubOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(SubKernel);
}

REGISTER_CPU_OPERATOR(Sub, ElementwiseSubOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Sub)
  .NumInputs(2)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
a - b = c
  )DOC");

template <typename DType, class Context>
bool MulKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Mul, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }

  VML_Mul<DType, CPUContext>(params.z_n, params.x, params.y, params.z, nullptr); 
  return true;
}

template <>
bool ElementwiseMulOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MulKernel);
}

REGISTER_CPU_OPERATOR(Mul, ElementwiseMulOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Mul)
  .NumInputs(2)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
a * b = c
  )DOC");

template <typename DType, class Context>
bool DivKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Div, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  VML_Div<DType, CPUContext>(params.z_n, params.x, params.y, params.z, nullptr); 
  return true;
}

template <>
bool ElementwiseDivOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(DivKernel);
}

REGISTER_CPU_OPERATOR(Div, ElementwiseDivOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Div)
  .NumInputs(2)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
a / b = c
  )DOC");

template <typename DType, class Context>
bool EqualKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Equal, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  for (size_t i = 0; i < params.z_n; ++i) {
    params.z[i] = params.x[i % params.x_n] == params.y[i % params.y_n] ? 1 : 0; 
  }
  return true;
}

template <>
bool ElementwiseEqualOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(EqualKernel);
}

REGISTER_CPU_OPERATOR(Equal, ElementwiseEqualOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Equal)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
c = (a == b)
  )DOC");

template <typename DType, class Context>
bool NotEqualKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::NotEqual, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  for (size_t i = 0; i < params.z_n; ++i) {
    params.z[i] = params.x[i % params.x_n] == params.y[i % params.y_n] ? 0 : 1; 
  }
  return true;
}

template <>
bool ElementwiseNotEqualOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(NotEqualKernel);
}

REGISTER_CPU_OPERATOR(NotEqual, ElementwiseNotEqualOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(NotEqual)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
c = (a != b)
  )DOC");

template <typename DType, class Context>
bool MaxKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Max, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  for (size_t i = 0; i < params.z_n; ++i) {
    params.z[i] = std::max(params.x[i % params.x_n], params.y[i % params.y_n]); 
  }
  return true;
}

template <>
bool ElementwiseMaxOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MaxKernel);
}

REGISTER_CPU_OPERATOR(Max, ElementwiseMaxOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Max)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
c = (a > b ? a : b)
  )DOC");

template <typename DType, class Context>
bool MinKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Min, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  for (size_t i = 0; i < params.z_n; ++i) {
    params.z[i] = std::min(params.x[i % params.x_n], params.y[i % params.y_n]); 
  }
  return true;
}

template <>
bool ElementwiseMinOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(MinKernel);
}

REGISTER_CPU_OPERATOR(Min, ElementwiseMinOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(Min)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
c = (a > b ? b : a)
  )DOC");

template <typename DType, class Context>
bool BroadcastToKernel(ElementwiseParam<DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<DType, broadcast::Assign, CpuKernelLauncher, CPUContext>(
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context); 
    return res; 
  }
  
  for (size_t i = 0; i < params.z_n; ++i) {
    params.z[i] = params.x[i % params.x_n]; 
  }
  return true;
}

template <>
bool BroadcastToOp<CPUContext>::RunOnDevice() {
  RUN_ELEMENTWISE_OP_KERNEL(BroadcastToKernel);
}

REGISTER_CPU_OPERATOR(BroadcastTo, BroadcastToOp<CPUContext>);

// Input: a, b Output: c
OPERATOR_SCHEMA(BroadcastTo)
  .NumInputs(1, 2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}, {1, 0}})
  .SetAttr<bool>(kAttrIsElementWise, true)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  })
  .SetDoc(R"DOC(
c = BroadcastTo(a)
  )DOC");

template <typename IType, typename DType, class Context>
static bool WhereKernel(TernaryElementwiseParam<IType, DType>& params, bool need_broadcast, const Context& context) {
  if (need_broadcast) {
    bool res = broadcast::BroadcastCompute<IType, DType, broadcast::Where, CpuKernelLauncher, CPUContext>(
        params.condition, params.condition_shape,
        params.x, params.x_shape, params.y, params.y_shape,
        params.z, params.z_shape, context);
    return res;
  }


  VML_Where<IType, DType, CPUContext>(params.z_n, params.condition, params.x, params.y, params.z, nullptr);
  return true;
}

template <>
bool WhereOp<CPUContext>::RunOnDevice() {
  Blob* condition = this->Input(0);
  Blob* x = this->Input(1);
  Blob* y = this->Input(2);
  Blob* z = this->Output(0);

  CheckValid();
  Reshape();
  bool need_broadcast = NeedBroadcast();

  ID_TYPE_SWITCH(condition->data_type(), IType, {
  TYPE_SWITCH(x->data_type(), DType, {
    TernaryElementwiseParam<IType, DType> params(condition->as<IType>(), condition->size(), condition->shape(),
                                                 x->as<DType>(), x->size(), x->shape(),
                                                 y->as<DType>(), y->size(), y->shape(),
                                                 z->as<DType>(), z->size(), z->shape());
    WhereKernel<IType, DType, CPUContext>(params, need_broadcast, this->context_);
  });
  });

  return true;
}

REGISTER_CPU_OPERATOR(Where, WhereOp<CPUContext>);

// Input: condition X Y Output: Z
OPERATOR_SCHEMA(Where)
        .NumInputs(3)
        .NumOutputs(1)
        .IdenticalTypeOfInput(1)
        .SetDoc(R"DOC(
Return elements, either from X or Y, depending on condition
  )DOC")
        .Input(0, "condition", "When True (nonzero), yield X, otherwise yield Y")
        .Input(1, "X", "N-D Input tensor")
        .Input(2, "Y", "N-D Input tensor")
        .Output(0, "Z", "N-D output tensor");

#undef RUN_ELEMENTWISE_OP_KERNEL

}  // namespace blaze

