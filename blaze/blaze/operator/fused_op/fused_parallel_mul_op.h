/*
 * \file fused_parallel_gemm_op.h 
 * \brief The fused parallel mul
 *
 * Such as:
 *
 *     (A, B) (A, B) (A, B)
 *      |      |      |
 *      |      |      |
 *      Mul   Mul    Mul
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "blaze/math/broadcast.h"

namespace blaze {

template <class Context>
class FusedParallelMulOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FusedParallelMulOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    parallel_num_ = OperatorBase::GetSingleArgument<int>("parallel_num", 2);
  }

  bool RunOnDevice() override {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    // Step1: Setup
    Setup();

    // Step2: Run fused Mul
    std::vector<TIndex> a_shape = a->shape();
    std::vector<TIndex> b_shape = b->shape();
    std::vector<TIndex> c_shape = c->shape();
    a_shape[0] /= parallel_num_;
    b_shape[0] /= parallel_num_;
    c_shape[0] /= parallel_num_;

    TIndex lda = 1, ldb = 1, ldc = 1;
    for (auto dim : a_shape) lda *= dim;
    for (auto dim : b_shape) ldb *= dim;
    for (auto dim : c_shape) ldc *= dim;

    TYPE_SWITCH_WITH_CTX(this->context_, a->data_type(), DType, {
    // Run Batched Mul
    DType* a0 = a->as<DType>();
    DType* b0 = b->as<DType>();
    DType* c0 = c->as<DType>();

    BatchedBroadcastMul<DType, Context>(a0,
                                        lda,
                                        a_shape,
                                        b0,
                                        ldb,
                                        b_shape,
                                        c0,
                                        ldc,
                                        c_shape,
                                        parallel_num_,
                                        &this->context_);
    });
    return true;
  }

 private:
  void Setup() {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    BLAZE_CONDITION_THROW(a->shape().size() >= 2, "a->shape().size()=",
                          a->shape().size());
    BLAZE_CONDITION_THROW(b->shape().size() >= 2, "b->shape().size()=",
                          b->shape().size());

    std::vector<TIndex> a_shape = a->shape();
    std::vector<TIndex> b_shape = b->shape();

    a_shape[0] /= parallel_num_;
    b_shape[0] /= parallel_num_;
    std::vector<TIndex> c_shape;

    bool success = MBroadcasting::BroadcastShape(a_shape, b_shape, c_shape);
    BLAZE_CONDITION_THROW(success, "can not broadcast for Mul");

    c_shape[0] *= parallel_num_;
    c->Reshape(c_shape);
  }

  int parallel_num_;
};

}  // namespace blaze
