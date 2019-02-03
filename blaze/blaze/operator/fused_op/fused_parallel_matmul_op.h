/*
 * \file fused_parallel_matmul_op.h 
 * \brief The fused parallel matmul
 *
 * Such as:
 *
 *    (A_1, B_1)  (A_2, B_2) (A_3, B_3)
 *      |            |          |
 *      |            |          |
 *      MatMul     MatMul     MatMul
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/broadcast.h"
#include "blaze/math/gemm.h"

namespace blaze {

template <class Context>
class FusedParallelMatMulOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FusedParallelMatMulOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    parallel_num_ = OperatorBase::GetSingleArgument<int>("parallel_num", 2);
    from_deepnet_ = OperatorBase::GetSingleArgument<bool>("from_deepnet", false);
    transa_ = OperatorBase::GetSingleArgument<bool>("transA", false);
    transb_ = OperatorBase::GetSingleArgument<bool>("transB", false);
  }

  bool RunOnDevice() override {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    // Step1: Setup
    Setup();

    // Step2: Run fused MatMul
    std::vector<TIndex> a_shape = a->shape();
    std::vector<TIndex> b_shape = b->shape();
    
    a_shape[0] /= parallel_num_;
    b_shape[0] /= parallel_num_;

     // Calc matrix M/K/N
    int M = a_shape[a_shape.size() - 2];
    int K = a_shape[a_shape.size() - 1];
    if (transa_) std::swap(M, K);
    int N = b_shape[b_shape.size() - 1];
    if (transb_) N = b_shape[b_shape.size() - 2];

    a_shape.resize(a_shape.size() - 2);
    b_shape.resize(b_shape.size() - 2);
    if (b_shape.size() < a_shape.size()) {
      for(size_t k = 0; k < a_shape.size() - b_shape.size(); ++k) {
        b_shape.insert(b_shape.begin(), 1);
      }
    } else {
      for (size_t k = 0; k < b_shape.size() - a_shape.size(); ++k) {
        a_shape.insert(a_shape.begin(), 1);
      }
    }

    TYPE_SWITCH_WITH_CTX(this->context_, a->data_type(), DType, {
    // Run StrideBatched
    DType* a0 = a->as<DType>();
    DType* b0 = b->as<DType>();
    DType* c0 = c->as<DType>();

    // Batched broadcastG GEMM
    BatchedBroadcastGemm<DType, Context>(transa_ ? CblasTrans : CblasNoTrans,
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
                                         parallel_num_,
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
      // ulf compatile process
      UlfCompatileProcess();
    }
    BLAZE_CONDITION_THROW(a->shape().size() >= 2, "a->shape().size()=", a->shape().size());
    BLAZE_CONDITION_THROW(b->shape().size() >= 2, "b->shape().size()=", b->shape().size());

    Blob *min = a, *max = b;
    bool min_trans = transa_, max_trans = transb_;
    if (a->shape().size() > b->shape().size()) {
      std::swap(min, max);
      std::swap(min_trans, max_trans);
    }

    std::vector<TIndex> min_shape = min->shape();
    std::vector<TIndex> max_shape = max->shape();
    
    BLAZE_CONDITION_THROW(min_shape[0] % parallel_num_ == 0,
                          "min_shape[0]=", min_shape[0], " parallel_num=", parallel_num_);
    BLAZE_CONDITION_THROW(max_shape[0] % parallel_num_ == 0,
                          "max_shape[0]=", max_shape[0], " parallel_num=", parallel_num_);

    min_shape[0] /= parallel_num_;
    max_shape[0] /= parallel_num_;

    std::vector<TIndex> shape = max_shape;
    size_t off = max_shape.size() - min_shape.size();
    for (size_t k = 0; k < min_shape.size() - 2; ++k) {
      TIndex dim_min = min_shape[k];
      TIndex dim_max = max_shape[off + k];
      BLAZE_CONDITION_THROW(dim_min == dim_max || dim_min == 1 || dim_max == 1,
                            "dim_min=", dim_min, " dim_max=", dim_max);
      if (dim_max == 1) {
        shape[off + k] = dim_min;
      }
    }

    // check matrix multiply validity
    TIndex a_k = a == min ? min_shape[min_shape.size() - 1] : max_shape[max_shape.size() - 1];
    if (transa_) {
      a_k = a == min ? min_shape[min_shape.size() - 2] : max_shape[max_shape.size() - 2];
    }
    TIndex b_k = b == min ? min_shape[min_shape.size() - 2] : max_shape[min_shape.size() - 2];
    if (transb_) {
      b_k = b == min ? min_shape[min_shape.size() - 1] : max_shape[max_shape.size() - 1];
    }
    BLAZE_CONDITION_THROW(a_k == b_k, "a_k=", a_k, " b_k=", b_k);

    if (transa_) {
      shape[shape.size() - 2] = min == a ? min_shape[min_shape.size() - 1] : max_shape[max_shape.size() - 1];
    } else {
      shape[shape.size() - 2] = min == a ? min_shape[min_shape.size() - 2] : max_shape[max_shape.size() - 2];
    }

    if (transb_) {
      shape[shape.size() - 1] = min == b ? min_shape[min_shape.size() - 2] : max_shape[max_shape.size() - 2];
    } else {
      shape[shape.size() - 1] = min == b ? min_shape[min_shape.size() - 1] : max_shape[max_shape.size() - 1];
    }

    // Reshape c.
    shape[0] *= parallel_num_;
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
  int parallel_num_;
  bool from_deepnet_;
};

}  // namespace blaze
