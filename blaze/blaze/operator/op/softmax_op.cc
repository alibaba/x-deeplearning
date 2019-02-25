/*
 * \file softmax_op.cc
 * \brief The softmax operation on CPU
 */
#include "blaze/operator/op/softmax_op.h"

#include <omp.h>
#include <math.h>

#include "blaze/math/vml.h"

namespace blaze {

template <>
SoftmaxOp<CPUContext>::SoftmaxOp(const OperatorDef& def, Workspace* workspace) :
    Operator<CPUContext>(def, workspace) {
  axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 1);
  iblob_.reset(new Blob(this->device_option_));
}

template <typename DType>
static void LauchKernel(Blob* X, Blob* Y, Blob* iblob, int axis) {
  size_t X_SIZE = X->size();
  size_t N = X->size(0, axis);
  size_t C = X->shape()[axis];
  size_t W = X->size(axis + 1, X->shape().size());

  iblob->Reshape({ N, W });
  
  // Calculate max values
  for (size_t i = 0; i < N; ++i) {
    size_t dst_offset = i * W;
    size_t src_offset = i * C * W;
    for (size_t z = 0; z < W; ++z) {
      iblob->as<DType>()[dst_offset + z] = X->as<DType>()[src_offset + z]; 
    }
    for (size_t j = 1; j < C; ++j) {
       src_offset += W;
       for (size_t z = 0; z < W; ++z) {
         iblob->as<DType>()[dst_offset + z] = std::max(iblob->as<DType>()[dst_offset + z],
                                                       X->as<DType>()[src_offset + z]);
       }
    }
  }

  // Sub max
  size_t Z = C * W;
  for (size_t i = 0; i < X_SIZE; ++i) {
    size_t index = i / Z;
    size_t offset = i % W;
    Y->as<DType>()[i] = X->as<DType>()[i] - iblob->as<DType>()[index * W + offset]; 
  }

  // Exp
  VML_Exp<DType, CPUContext>(Y->size(), Y->as<DType>(), Y->as<DType>(), nullptr);

  // Sum
  for (size_t i = 0; i < N; ++i) {
    size_t offset = i * W;
    size_t src_offset = i * C * W;
    for (size_t k = 0; k < W; ++k) {
      iblob->as<DType>()[offset + k] = 0.0;
    }
    for (size_t j = 0; j < C; ++j) {
      for (size_t k = 0; k < W; ++k) {
        iblob->as<DType>()[offset + k] += Y->as<DType>()[src_offset++];
      }
    }
    offset += W;
  }

  // Div
  for (size_t k = 0; k < X_SIZE; ++k) {
    size_t index = k / Z;
    size_t offset = k % W;
    Y->as<DType>()[k] /= iblob->as<DType>()[index * W + offset];
  }
}

template <>
bool SoftmaxOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);
  
  TYPE_SWITCH(X->data_type(), DType, {
  
  // Reshape
  Y->Reshape(X->shape());

  iblob_->set_data_type(static_cast<DataType>(X->data_type()));
  // Launch kernel
  LauchKernel<DType>(X, Y, iblob_.get(), axis_);

  });

  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Softmax)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .SetDoc(R"DOC(
Softamx activation
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D Output tensor");

}  // namespace blaze

