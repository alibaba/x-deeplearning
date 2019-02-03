/*
 * \file fused_parallel_mul_reducesum_op.h
 * \file The fused parallel mul reducesum operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include <sstream>

#include "blaze/common/proto_helper.h"
#include "blaze/operator/op/reduce_sum_op.h"
#include "blaze/operator/fused_op/fused_parallel_mul_op.h"

namespace blaze {

template <typename Context>
class FusedParallelMulReduceSumOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FusedParallelMulReduceSumOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { 
    if (OperatorBase::HasArgument("axes")) {
      std::vector<size_t> axes = OperatorBase::GetRepeatedArgument<size_t>("axes");
      BLAZE_CONDITION_THROW(axes.size() == 1, "axes.size()=", axes.size());
      axis_ = axes[0];
    } else {
      axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 0);
    }
    keepdims_ = OperatorBase::GetSingleArgument<int>("keepdims", 1);
    parallel_num_ = OperatorBase::GetSingleArgument<int>("parallel_num", 2);

    CreateParallelMulOp(def, workspace);
    CreateReduceOp(def, workspace);
  }

  bool RunOnDevice() override {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    std::vector<TIndex> a_shape = a->shape();
    std::vector<TIndex> b_shape = b->shape();
    a_shape[0] /= parallel_num_;
    b_shape[0] /= parallel_num_;
    
    std::vector<TIndex> c_shape;
    bool success = MBroadcasting::BroadcastShape(a_shape, b_shape, c_shape);
    BLAZE_CONDITION_THROW(success, "can not broadcast for ParallelMulReduce");
    
    if (CanFusionExec(a_shape, b_shape, c_shape)) {
      std::vector<TIndex> new_c_shape;
      for (size_t i = 0; i < c_shape.size(); ++i) {
        if (axis_ == i) {
          if (keepdims_) new_c_shape.push_back(1);
        } else { new_c_shape.push_back(c_shape[i]); }
      }
      new_c_shape[0] *= parallel_num_;
      c->Reshape(new_c_shape);

      TYPE_SWITCH_WITH_CTX(this->context_, c->data_type(), DType, {
        TIndex batch_count = parallel_num_;
        TIndex M = a_shape[0] == 1 ? b_shape[0] : a_shape[0];
        TIndex K = a_shape[0] == 1 ? b_shape[1] : a_shape[1];
        TIndex N = a_shape[0] == 1 ? a_shape[2] : b_shape[2];
        DType* a_dptr = a_shape[0] == 1 ? b->as<DType>() : a->as<DType>();
        DType* b_dptr = a_shape[0] == 1 ? a->as<DType>() : b->as<DType>();
        DType* c_dptr = c->as<DType>();
        GemmStridedBatched<DType, Context>(CblasNoTrans,
                                           CblasNoTrans,
                                           M,
                                           N,
                                           K,
                                           1.0,
                                           a_dptr,
                                           M * K,
                                           b_dptr,
                                           K *N,
                                           0,
                                           c_dptr,
                                           M * N,
                                           batch_count,
                                           &this->context_); 
      });
    } else {
      fused_parallel_mul_op_->Run(this->stream_id());
     
      // ReduceSum not support parallel operation.
      Blob* input = reduce_op_->Input(0);
      std::vector<TIndex> shp = input->shape();
      shp[0] /= parallel_num_;
      shp.insert(shp.begin(), parallel_num_);
      input->Reshape(shp);

      reduce_op_->Run(this->stream_id());

      Blob* output = reduce_op_->Output(0);
      shp = output->shape();
      shp[0] *= shp[1];
      for (size_t k = 1; k + 1 < shp.size(); ++k) {
        shp[k] = shp[k + 1];
      }
      shp.resize(shp.size() - 1);
      output->Reshape(shp);
    }
    return true;
  }

 protected:
  virtual void ResetEvent() {
    OperatorBase::ResetEvent();
    fused_parallel_mul_op_->ResetEvent();
    reduce_op_->ResetEvent();
  }
  void CreateParallelMulOp(const OperatorDef& def, Workspace* workspace) {
    OperatorDef temp_def = def;

    temp_def.set_name("");
    temp_def.set_type("FusedParallelMul");

    const std::string& oname = temp_def.output(0);
    std::stringstream ss;
    ss << oname << "_0";
    temp_def.clear_output();
    temp_def.add_output(ss.str());

    fused_parallel_mul_op_.reset(new FusedParallelMulOp<Context>(temp_def, workspace));
  }
  void CreateReduceOp(const OperatorDef& def, Workspace* workspace) {
    OperatorDef temp_def = def;

    temp_def.set_name("");
    temp_def.set_type("ReduceSum");

    const std::string& oname = temp_def.output(0);
    std::stringstream ss;
    ss << oname << "_0";
    temp_def.clear_input();
    temp_def.add_input(ss.str());

    // ReduceSum not support Parallel operation, so inc axis.
    if (OperatorBase::HasArgument("axes")) {
      ArgumentHelper::SetRepeatedArgument<size_t>(temp_def, "axis", { axis_ + 1 });
    } else {
      ArgumentHelper::SetSingleArgument<size_t>(temp_def, "axis", axis_ + 1);
    }

    reduce_op_.reset(new ReduceSumOp<Context>(temp_def, workspace));
  }

  void CheckValid() {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);

    const std::vector<TIndex>& shape = x->shape();
    BLAZE_CONDITION_THROW(axis_ < shape.size(),
                          "axis_=", axis_,
                          " shape.size()=", shape.size());
  }

  bool CanFusionExec(std::vector<TIndex>& x_shape,
                     std::vector<TIndex>& y_shape,
                     std::vector<TIndex>& c_shape) {
    if (c_shape.size() != 3 || axis_ != 1) return false;
    if (x_shape.size() <= 3 && y_shape.size() <= 3) {
      for (TIndex k = x_shape.size(); k < 3; ++k) x_shape.insert(x_shape.begin(), 1);
      for (TIndex k = y_shape.size(); k < 3; ++k) y_shape.insert(y_shape.begin(), 1);
      if (x_shape[1] == y_shape[1]) {
        if (x_shape[0] < y_shape[0] && x_shape[2] > y_shape[2]) return true;
        if (x_shape[0] > y_shape[0] && x_shape[2] < y_shape[2]) return true;
      }
    }
    return false;
  }

  int parallel_num_;
  size_t axis_;
  int keepdims_;

  std::shared_ptr<OperatorBase> fused_parallel_mul_op_;
  std::shared_ptr<OperatorBase> reduce_op_;
};

}  // namespace blaze
