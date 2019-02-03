/*
 * \file gemm_op.h 
 * \brief The gemm operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/broadcast.h"
#include "blaze/math/vml.h"
#include "blaze/math/gemm.h"

namespace blaze {

template <class Context>
class GemmOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  GemmOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    // Unlike ONNX, should support 3D FullyConnect operation.
    transa_ = OperatorBase::GetSingleArgument<bool>("transA", false);
    transb_ = OperatorBase::GetSingleArgument<bool>("transB", false);
    alpha_ = OperatorBase::GetSingleArgument<float>("alpha", 1.0);
    beta_ = OperatorBase::GetSingleArgument<float>("beta", 1.0);
  }

  bool RunOnDevice() override {
    CheckValid();
    
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->InputSize() > 2 ? this->Input(2) : nullptr;
    Blob* y = this->Output(0);

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    // Reshape
    TIndex M = a_shape.size() == 3 ? a_shape[1] : a_shape[0];
    TIndex K = a_shape.size() == 3 ? a_shape[2] : a_shape[1];
    if (transa_) {
      std::swap(M, K);
    }
    TIndex N = b_shape[1];
    if (transb_) N = b_shape[0];
    bool is_3dgemm = false;
    if (a_shape.size() == 3) {
      y->Reshape({ a_shape[0], M, N });  // 3D Gemm
      if (M == 1) {  // we can use 2D Gemm yet
        M = a_shape[0];
      } else {
        is_3dgemm = true;
      }
    } else {
      y->Reshape({ M, N });
    }

    // Calculate
    float beta = beta_;
    // Step1: Y = C
    if (c == nullptr) {
      beta = 0;
    } else {
      TYPE_SWITCH_WITH_CTX(this->context_, y->data_type(), DType, {
        DimEqualBroadcastAssign<DType, Context>(y->as<DType>(), y->shape(),
                                                c->as<DType>(), c->shape(), &this->context_);          
      });
    }
    // Step2: Y = alpha * A * B + beta * C
    TYPE_SWITCH_WITH_CTX(this->context_, a->data_type(), DType, {
      if (is_3dgemm) {
        GemmStridedBatched<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                                           transb_ ? CblasTrans : CblasNoTrans,
                                           M,
                                           N,
                                           K,
                                           alpha_,
                                           a->as<DType>(),
                                           M * K,
                                           b->as<DType>(),
                                           0,
                                           beta,
                                           y->as<DType>(),
                                           M * N,
                                           a_shape[0],
                                           &this->context_);
      } else {
        Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                             transb_ ? CblasTrans : CblasNoTrans,
                             M,
                             N,
                             K,
                             alpha_,
                             a->as<DType>(),
                             b->as<DType>(),
                             beta,
                             y->as<DType>(),
                             &this->context_);
      }
    });
    return true;
  }

 protected:
  void CheckValid() {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->InputSize() > 2 ? this->Input(2) : nullptr;
    Blob* y = this->Output(0);

    BLAZE_CONDITION_THROW(a->shape().size() == 2 || a->shape().size() == 3,
                          "a->shape.size()=", a->shape().size());
    BLAZE_CONDITION_THROW(b->shape().size() == 2,
                          "b->shape.size()=", b->shape().size(), this->def_.DebugString());
    if (c != nullptr) {
      BLAZE_CONDITION_THROW(c->shape().size() <= 2,
                            "c->shape().size()=", c->shape().size());
    }
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    TIndex a_m = a_shape.size() == 3 ? a_shape[1] : a_shape[0];
    TIndex a_k = a_shape.size() == 3 ? a_shape[2] : a_shape[1];
    if (transa_) {
      std::swap(a_m, a_k);
    }
    TIndex b_k = b_shape[0];
    TIndex b_n = b_shape[1];
    if (transb_) {
      std::swap(b_k, b_n);
    }

    BLAZE_CONDITION_THROW(a_k == b_k, "a_k=", a_k, " b_k=", b_k, this->def_.DebugString());

    if (c != nullptr) {
      // Must unibroacsting and DimEqual
      if (c->shape().size() == 2) {
        BLAZE_CONDITION_THROW(a_m == c->shape()[0], "a_m=", a_m, " c->shape()[0]=", c->shape()[0]);
        BLAZE_CONDITION_THROW(b_n == c->shape()[1], "b_n=", b_n, " c->shape()[1]=", c->shape()[1]);
      } else {
        BLAZE_CONDITION_THROW(b_n == c->shape()[0], "b_n=", b_n, " c->shape()[0]=", c->shape()[0]);
      }
    }
  }

  float alpha_;
  float beta_;
  bool transa_;
  bool transb_;
};

}  // namespace blaze

