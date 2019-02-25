/*
 * \file combine_func_op.h
 * \brief The combine func operator for combine result processing.
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"
#include "blaze/operator/sparse_op/cross/cross_common_defines.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
struct CombineFuncParam {
  K_DType* input_ids;
  V_DType* input_values;
  N_DType* input_nums;
  K_DType* output_ids;
  V_DType* output_values;
  N_DType* output_nums;
  size_t id_size;
  size_t num_size;
  Blob* output_id_blob;
  Blob* output_value_blob;
};

template <class Context>
class CombineFuncOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  CombineFuncOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    int func_type = OperatorBase::GetSingleArgument<int>("func_type", kCombineProcessFuncIDFlag);
    combine_process_func_type_ = static_cast<CombineProcessFuncType>(func_type);
    reduce_feature_id_ =
        OperatorBase::GetSingleArgument<int64_t>("reduce_feature_id", 0);
  }

  bool RunOnDevice() override;

 protected:
  bool IsReduceCombiner() {
    switch (combine_process_func_type_) {
      case kCombineProcessFuncIDFlag:
      case kCombineProcessFuncLogIDFlag:
        return false;
      case kCombineProcessFuncSumFlag:
      case kCombineProcessFuncLogSumFlag:
      case kCombineProcessFuncMaxFlag:
      case kCombineProcessFuncLogMaxFlag:
      case kCombineProcessFuncMinFlag:
      case kCombineProcessFuncLogMinFlag:
      case kCombineProcessFuncAvgFlag:
      case kCombineProcessFuncLogAvgFlag:
      case kCombineProcessFuncCosFlag:
      case kCombineProcessFuncLogCosFlag:
      case kCombineProcessFuncDotSumFlag:
      case kCombineProcessFuncLogDotSumFlag:
      case kCombineProcessFuncDotL1NormFlag:
      case kCombineProcessFuncDotL2NormFlag:
        return true;
      default:
        BLAZE_THROW("unexpected combine func type", combine_process_func_type_);
    }
  }

  // Prepare combine func param
  template <typename K_DType, typename V_DType, typename N_DType>
  void Setup(CombineFuncParam<K_DType, V_DType, N_DType>* params) {
    // prepare input params
    Blob* input_id_blob = this->Input(0);
    Blob* input_value_blob = this->Input(1);
    Blob* input_num_blob = this->Input(2);
    params->input_ids = input_id_blob->as<K_DType>();
    params->input_values = input_value_blob->as<V_DType>();
    params->input_nums = input_num_blob->as<N_DType>();
    params->id_size = input_id_blob->size();
    params->num_size = input_num_blob->size();

    // prepare output params
    Blob* output_id_blob = this->Output(0);
    Blob* output_value_blob = this->Output(1);
    Blob* output_num_blob = this->Output(2);
    if (IsReduceCombiner()) {  // reduce combiner: n input -> n output
      const auto& output_shape = input_num_blob->shape();
      output_id_blob->Reshape(output_shape);
      output_value_blob->Reshape(output_shape);
      output_num_blob->Reshape(output_shape);
    } else {  // element-wise combiner: n input -> n output
      const auto& output_id_shape = input_id_blob->shape();
      const auto& output_num_shape = input_num_blob->shape();
      output_id_blob->Reshape(output_id_shape);
      output_value_blob->Reshape(output_id_shape);
      output_num_blob->Reshape(output_num_shape);
    }

    params->output_ids = output_id_blob->as<K_DType>();
    params->output_values = output_value_blob->as<V_DType>();
    params->output_nums = output_num_blob->as<N_DType>();
    params->output_id_blob = output_id_blob;
    params->output_value_blob = output_value_blob;
  }

  void CheckValid() {
    Blob* input_id = this->Input(0);
    Blob* input_value = this->Input(1);
    Blob* input_num = this->Input(2);

    // id & value size must be equal
    BLAZE_CONDITION_THROW(input_id->size() == input_value->size(),
                          "input_id->size()=",
                          input_id->size(),
                          " input_value->size()=",
                          input_value->size());
  }

  int64_t reduce_feature_id_;
  CombineProcessFuncType combine_process_func_type_;
};

}  // namespace blaze
