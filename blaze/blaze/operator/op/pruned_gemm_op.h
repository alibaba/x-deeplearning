/*
 * \file pruned_gemm_op.h
 * \desc The pruned gemm operator
 * 
 *  A | B  x     C     = A x C + B x D
 *               --
 *               D 
 *
 *  Only support 2D-Gemm
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/broadcast.h"
#include "blaze/math/vml.h"
#include "blaze/math/gemm.h"

namespace blaze {

template <class Context>
class PrunedGemmOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  PrunedGemmOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    transa_ = OperatorBase::GetSingleArgument<bool>("transA", false);
    transb_ = OperatorBase::GetSingleArgument<bool>("transB", false);
    alpha_ = OperatorBase::GetSingleArgument<float>("alpha", 1.0);
    beta_ = OperatorBase::GetSingleArgument<float>("beta", 1.0);

    iblob_.reset(new Blob(this->device_option_));
  }

  bool RunOnDevice() override {
    CheckValid();

    Blob* x1 = this->Input(0);
    Blob* x2 = this->Input(1);
    Blob* w1 = this->Input(2);
    Blob* w2 = this->Input(3);
    Blob* bias = this->InputSize() > 4 ? this->Input(4) : nullptr;
    Blob* y = this->Output(0);

    const auto& x1_shape = x1->shape();
    const auto& x2_shape = x2->shape();
    const auto& w1_shape = w1->shape();
    const auto& w2_shape = w2->shape();
    // Reshape
    TIndex x1_m = x1_shape[0];
    TIndex x1_k = x1_shape[1];
    if (transa_) std::swap(x1_m, x1_k);
    TIndex x2_m = x2_shape[0];
    TIndex x2_k = x2_shape[1];
    if (transb_) std::swap(x2_m, x2_k);

    TIndex w1_n = w1_shape[1];
    if (transb_) w1_n = w1_shape[0];
    TIndex w2_n = w1_n;

    y->Reshape({ std::max(x1_m, x2_m), w1_n }); 

    // Calculate
    float beta = beta_;
    // Step1: Y = bias
    if (bias == nullptr) {
      beta = 0;
    } else {
      TYPE_SWITCH_WITH_CTX(this->context_, y->data_type(), DType, {
        DimEqualBroadcastAssign<DType, Context>(y->as<DType>(), y->shape(),
                                                bias->as<DType>(), bias->shape(), &this->context_);
      });
    }

    iblob_->set_data_type(static_cast<DataType>(y->data_type()));
    iblob_->Reshape({ w1_n });

    // Step2: Y = alpha * X1 * W1 + alpha * X2 * W2 + beta * Y
    TYPE_SWITCH_WITH_CTX(this->context_, y->data_type(), DType, {
      if (x1_m == x2_m) {
        Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                             transb_ ? CblasTrans : CblasNoTrans,
                             x1_m,
                             w1_n,
                             x1_k,
                             alpha_,
                             x1->as<DType>(),
                             w1->as<DType>(),
                             beta,
                             y->as<DType>(),
                             &this->context_);
        Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                             transb_ ? CblasTrans : CblasNoTrans,
                             x2_m,
                             w2_n,
                             x2_k,
                             alpha_,
                             x2->as<DType>(),
                             x2->as<DType>(),
                             beta,
                             y->as<DType>(),
                             &this->context_);
      } else {
        if (x1_m == 1) {
          Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                               transb_ ? CblasTrans : CblasNoTrans,
                               x2_m,
                               w2_n,
                               x2_k,
                               alpha_,
                               x2->as<DType>(),
                               w2->as<DType>(),
                               beta,
                               y->as<DType>(),
                               &this->context_);
          Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                               transb_ ? CblasTrans : CblasNoTrans,
                               x1_m,
                               w1_n,
                               x1_k,
                               alpha_,
                               x1->as<DType>(),
                               w1->as<DType>(),
                               0,
                               iblob_->as<DType>(),
                               &this->context_);
        } else {  // x2_m == 1
          Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                               transb_ ? CblasTrans : CblasNoTrans,
                               x1_m,
                               w1_n,
                               x1_k,
                               alpha_,
                               x1->as<DType>(),
                               w1->as<DType>(),
                               beta,
                               y->as<DType>(),
                               &this->context_);
          Gemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                               transb_ ? CblasTrans : CblasNoTrans,
                               x2_m,
                               w2_n,
                               x2_k,
                               alpha_,
                               x2->as<DType>(),
                               w2->as<DType>(),
                               0,
                               iblob_->as<DType>(),
                               &this->context_);
        }
        // Broadcast FMA
        DimEqualBroadcastFMA<DType, Context>(y->as<DType>(),
                                             y->shape(),
                                             iblob_->as<DType>(),
                                             iblob_->shape(),
                                             &this->context_);
      }
    });
    return true;
  }

 protected:
  void CheckValid() {
    Blob* x1 = this->Input(0);
    Blob* x2 = this->Input(1);
    Blob* w1 = this->Input(2);
    Blob* w2 = this->Input(3);
    Blob* bias = this->InputSize() > 4 ? this->Input(4) : nullptr;

    BLAZE_CONDITION_THROW(x1->shape().size() == 2, 
                          "x1->shape().size()=",
                          x1->shape().size());
    BLAZE_CONDITION_THROW(x2->shape().size() == 2,
                          "x2->shape().size()=",
                          x2->shape().size());
    if (bias != nullptr) {
      BLAZE_CONDITION_THROW(bias->shape().size() <= 2,
                            "bias->shape().size()=",
                            bias->shape().size());
    }
    const auto& x1_shape = x1->shape();
    const auto& x2_shape = x2->shape();
    const auto& w1_shape = w1->shape();
    const auto& w2_shape = w2->shape();

    // Step1: Check X1/X2
    TIndex x1_m = x1_shape[0];
    TIndex x1_k = x1_shape[1];
    if (transa_) std::swap(x1_m, x1_k);

    TIndex x2_m = x2_shape[0];
    TIndex x2_k = x2_shape[1];
    if (transa_) std::swap(x2_m, x2_k);

    BLAZE_CONDITION_THROW(x1_m == x2_m || x1_m == 1 || x2_m == 1,
                          "x1_m=", x1_m, " x2_m=", x2_m);

    // Step2: Check W1/W2
    TIndex w1_k = w1_shape[0];
    TIndex w1_n = w1_shape[1];
    if (transb_) std::swap(w1_k, w1_n);

    TIndex w2_k = w2_shape[0];
    TIndex w2_n = w2_shape[1];
    if (transb_) std::swap(w2_k, w2_n);

    BLAZE_CONDITION_THROW(x1_k == w1_k, "x1_k=", x1_k, " w1_k=", w1_k);
    BLAZE_CONDITION_THROW(x2_k == w2_k, "w2_k=", x2_k, " w2_k=", w2_k);
    BLAZE_CONDITION_THROW(w1_n == w2_n, "w1_n=", w1_n, " w2_n=", w2_n);

    // Step3: Check bias
    if (bias) {
      const std::vector<TIndex>& bias_shape = bias->shape();
      if (bias_shape.size() == 2) {
        BLAZE_CONDITION_THROW(x1_m == bias_shape[0] || x2_m == bias_shape[0],
                              "x1_m=", x1_m, " x2_m=", x2_m,
                              " bias_shape[0]=", bias_shape[0]);
        BLAZE_CONDITION_THROW(w1_n == bias_shape[1],
                              "w1_n=", w1_n, " bias_shape[1]=", bias_shape[1]);
      } else {
        BLAZE_CONDITION_THROW(w1_n == bias_shape[0],
                              "w1_n=", w1_n, " bias_shape[0]=", bias_shape[0]);
      }
    }
  }

  std::shared_ptr<Blob> iblob_;
  float alpha_;
  float beta_;
  float transa_;
  float transb_;
};

}  // namespace blaze
