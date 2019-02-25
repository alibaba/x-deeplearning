/*
 * \file matmul_op.h 
 * \brief The matmul operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/broadcast.h"
#include "blaze/math/gemm.h"
#include "blaze/math/vml.h"

namespace blaze {

template <class Context>
class MatMulOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  MatMulOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    transa_ = OperatorBase::GetSingleArgument<bool>("transA", false);
    transb_ = OperatorBase::GetSingleArgument<bool>("transB", false);
    from_deepnet_ = OperatorBase::GetSingleArgument<bool>("from_deepnet", false);
  }

  bool RunOnDevice() override {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    // Step1: Setup
    Setup();
    
    // Step2: Run MatMul
    std::vector<TIndex> a_shape, b_shape;
    
    // Calc outer shape
    for (size_t k = 0; k < a->shape().size() - 2; ++k) {
      a_shape.push_back(a->shape()[k]);
    }
    for (size_t k = 0; k < b->shape().size() - 2; ++k) {
      b_shape.push_back(b->shape()[k]);
    }
    if (b_shape.size() < a_shape.size()) {
      for (size_t k = 0; k < a_shape.size() - b_shape.size(); ++k) {
        b_shape.insert(b_shape.begin(), 1);
      }
    } else if (b_shape.size() > a_shape.size()) {
      for (size_t k = 0; k < b_shape.size() - a_shape.size(); ++k) {
        a_shape.insert(a_shape.begin(), 1);
      }
    }

    // Calc matrix M/K/N
    int M = a->shape()[a->shape().size() - 2];
    int K = a->shape()[a->shape().size() - 1];
    if (transa_) {
      std::swap(M, K);
    }
    int N = b->shape()[b->shape().size() - 1];
    if (transb_) {
      N = b->shape()[b->shape().size() - 2];
    }

    TYPE_SWITCH_WITH_CTX(this->context_, a->data_type(), DType, {
    // Run StrideBatched
    DType* a0 = a->as<DType>();
    DType* b0 = b->as<DType>();
    DType* c0 = c->as<DType>();

    // Broacast GEMM
    BroadcastGemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
                                  transb_ ? CblasTrans : CblasNoTrans,
                                  M,
                                  N,
                                  K,
                                  1.0,
                                  a0,
                                  b0,
                                  0,
                                  c0,
                                  a_shape,
                                  b_shape,
                                  &this->context_);
    });
    return true;
  }

 protected:
  void Setup() {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    if (from_deepnet_) {
      // Ulf compatile process
      UlfCompatileProcess();
    }

    BLAZE_CONDITION_THROW(a->shape().size() >= 2,
                          "a->shape().size()=",
                          a->shape().size());
    BLAZE_CONDITION_THROW(b->shape().size() >= 2,
                          "b->shape().size()=",
                          b->shape().size());

    Blob *min = a, *max = b;
    bool min_trans = transa_, max_trans = transb_;
    if (a->shape().size() > b->shape().size()) {
      std::swap(min, max);
      std::swap(min_trans, max_trans);
    }

    std::vector<TIndex> shape = max->shape();
    size_t off = max->shape().size() - min->shape().size();
    for (size_t k = 0; k < min->shape().size() - 2; ++k) {
      TIndex dim_min = min->shape()[k];
      TIndex dim_max = max->shape()[off + k];
      // Broadcast support. numpy standard.
      BLAZE_CONDITION_THROW(dim_min == dim_max || dim_min == 1 || dim_max == 1,
                            "dim_min=", dim_min, " dim_max=", dim_max, " op=", this->def_.name());
      if (dim_max == 1) {
        shape[off + k] = dim_min;
      }
    }
    TIndex a_k = a->shape()[a->shape().size() - 1];
    if (transa_) a_k = a->shape()[a->shape().size() - 2];
    TIndex b_k = b->shape()[b->shape().size() - 2];
    if (transb_) b_k = b->shape()[b->shape().size() - 1];
    BLAZE_CONDITION_THROW(a_k == b_k, "a_k=", a_k, " b_k=", b_k);

    if (transa_) shape[shape.size() - 2] = a->shape()[a->shape().size() - 1];
    else shape[shape.size() - 2] = a->shape()[a->shape().size() - 2];
    
    if (transb_) shape[shape.size() - 1] = b->shape()[b->shape().size() - 2];
    else shape[shape.size() - 1] = b->shape()[b->shape().size() - 1];

    c->Reshape(shape);
  }

  // Ulf model compatible process
  void UlfCompatileProcess() {
    // According to the batchdot definition in ulf.
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    if (b->shape().size() == 2) {
      std::vector<TIndex> rb_shape(3);
      rb_shape[0] = b->shape()[0];
      rb_shape[1] = 1;
      rb_shape[2] = b->shape()[1];
      b->Reshape(rb_shape);
    }
    if (a->shape().size() == 2) {
      std::vector<TIndex> ra_shape(3);
      ra_shape[0] = a->shape()[0];
      ra_shape[1] = a->shape()[1];
      ra_shape[2] = 1;
      a->Reshape(ra_shape);
    }
  }
  
  bool transa_;
  bool transb_;
  bool from_deepnet_;
};

}  // namespace blaze

