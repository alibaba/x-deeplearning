/*
 * \file combine_func_op.cc
 * \brief The combine func operator for combine result processing.
 */
#include "blaze/operator/sparse_op/cross/combine_func_op.h"

namespace {
const float kSmallConst = 0.00000001;
}  // namespace

namespace blaze {

// ID combine function
template <typename K_DType, typename V_DType, typename N_DType>
void IdCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params) {
  if (params.output_ids != params.input_ids) {
    memcpy(params.output_ids, params.input_ids, sizeof(K_DType) * params.id_size);
  }
  if (params.output_values != params.input_values) {
    memcpy(params.output_values, params.input_values, sizeof(V_DType) * params.id_size);
  }
  if (params.output_nums != params.input_nums) {
    memcpy(params.output_nums, params.input_nums, sizeof(N_DType) * params.num_size);
  }
};

// LogID combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogIdCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                      N_DType input_start,
                      N_DType num,
                      N_DType output_start,
                      size_t output_num_index) {
  N_DType output_num = 0;
  for (auto i = input_start; i < input_start + num; ++i) {
    if (params.input_values[i] < 0)
      continue;
    params.output_ids[output_start + output_num] = params.input_ids[i];
    params.output_values[output_start + output_num] =
        log(params.input_values[i] + kSmallConst);
    output_num++;
  }
  params.output_nums[output_num_index] = output_num;
};

// Sum reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void SumCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    N_DType input_start,
                    N_DType num,
                    N_DType output_start,
                    size_t output_num_index,
                    K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    params.output_ids[output_start] = reduce_feature_id;
    params.output_values[output_start] = 0;
    for (auto i = input_start; i < input_start + num; ++i) {
      params.output_values[output_start] += params.input_values[i];
    }
    output_num = 1;
  }
  params.output_nums[output_num_index] = output_num;
};

// LogSum reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogSumCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    SumCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// Max reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void MaxCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    N_DType input_start,
                    N_DType num,
                    N_DType output_start,
                    size_t output_num_index,
                    K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    params.output_ids[output_start] = reduce_feature_id;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (i == input_start || params.output_values[output_start] < params.input_values[i]) {
        params.output_values[output_start] = params.input_values[i];
      }
    }
    output_num = 1;
  }
  params.output_nums[output_num_index] = output_num;
};

// LogMax reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogMaxCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    MaxCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// Min reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void MinCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    N_DType input_start,
                    N_DType num,
                    N_DType output_start,
                    size_t output_num_index,
                    K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    params.output_ids[output_start] = reduce_feature_id;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (i == input_start || params.output_values[output_start] > params.input_values[i]) {
        params.output_values[output_start] = params.input_values[i];
      }
    }
    output_num = 1;
  }
  params.output_nums[output_num_index] = output_num;
};

// LogMin reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogMinCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    MinCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// Avg reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void AvgCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    N_DType input_start,
                    N_DType num,
                    N_DType output_start,
                    size_t output_num_index,
                    K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    SumCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    params.output_ids[output_start] = reduce_feature_id;
    params.output_values[output_start] = 1.0 * params.output_values[output_start] / num;
    output_num = 1;
  }
  params.output_nums[output_num_index] = output_num;
};

// LogAvg reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogAvgCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    AvgCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// Cos reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void CosCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    N_DType input_start,
                    N_DType num,
                    N_DType output_start,
                    size_t output_num_index,
                    K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    bool contain_sum_aa = false;
    bool contain_sum_ab = false;
    bool contain_sum_bb = false;
    float sum_aa = 0.0f;
    float sum_ab = 0.0f;
    float sum_bb = 0.0f;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (params.input_ids[i] == kSumAaHashID) {
        contain_sum_aa = true;
        sum_aa = params.input_values[i];
      } else if (params.input_ids[i] == kSumAbHashID) {
        contain_sum_ab = true;
        sum_ab = params.input_values[i];
      } else if (params.input_ids[i] == kSumBbHashID) {
        contain_sum_bb = true;
        sum_bb = params.input_values[i];
      }
    }
    if (contain_sum_aa && contain_sum_ab && contain_sum_bb
        && sum_aa > 0 && sum_bb > 0) {
      params.output_ids[output_start] = reduce_feature_id;
      params.output_values[output_start] = sum_ab / (sqrt(sum_aa) * sqrt(sum_bb));
      output_num = 1;
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// LogCos reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogCosCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    CosCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                              output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// DotSum reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void DotSumCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                       N_DType input_start,
                       N_DType num,
                       N_DType output_start,
                       size_t output_num_index,
                       K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    bool contain_dot_sum = false;
    float sum_ab = 0.0f;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (params.input_ids[i] == kSumAbHashID) {
        contain_dot_sum = true;
        sum_ab = params.input_values[i];
        break;
      }
    }
    if (contain_dot_sum) {
      params.output_ids[output_start] = reduce_feature_id;
      params.output_values[output_start] = sum_ab;
      output_num = 1;
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// LogDotSum reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void LogDotSumCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                          N_DType input_start,
                          N_DType num,
                          N_DType output_start,
                          size_t output_num_index,
                          K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    DotSumCombineFunc<K_DType, V_DType, N_DType>(params, input_start, num, output_start,
                                                 output_num_index, reduce_feature_id);
    for (auto i = 0; i < params.output_nums[output_num_index]; ++i) {
      if (params.output_values[output_start + i] >= 0) {
        params.output_values[output_start + i] =
            log(params.output_values[output_start + i] + kSmallConst);
        output_num++;
      }
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// DotL1Norm reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void DotL1NormCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                          N_DType input_start,
                          N_DType num,
                          N_DType output_start,
                          size_t output_num_index,
                          K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    bool contain_dot_l1_norm = false;
    float sum_abs_ab = 0.0f;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (params.input_ids[i] == kSumAbsAbHashID) {
        contain_dot_l1_norm = true;
        sum_abs_ab = params.input_values[i];
        break;
      }
    }
    if (contain_dot_l1_norm) {
      params.output_ids[output_start] = reduce_feature_id;
      params.output_values[output_start] = sum_abs_ab;
      output_num = 1;
    }
  }
  params.output_nums[output_num_index] = output_num;
};

// DotL2Norm reduce combine function
template <typename K_DType, typename V_DType, typename N_DType>
void DotL2NormCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                          N_DType input_start,
                          N_DType num,
                          N_DType output_start,
                          size_t output_num_index,
                          K_DType reduce_feature_id) {
  N_DType output_num = 0;
  if (num > 0) {
    bool contain_dot_l2_norm = false;
    float sum_abab = 0.0f;
    for (auto i = input_start; i < input_start + num; ++i) {
      if (params.input_ids[i] == kSumAbAbHashID) {
        contain_dot_l2_norm = true;
        sum_abab = params.input_values[i];
        break;
      }
    }
    if (contain_dot_l2_norm) {
      params.output_ids[output_start] = reduce_feature_id;
      params.output_values[output_start] = sum_abab;
      output_num = 1;
    }
  }
  params.output_nums[output_num_index] = output_num;
};

template <typename K_DType, typename V_DType, typename N_DType>
void RunCombineFunc(CombineFuncParam<K_DType, V_DType, N_DType>& params,
                    CombineProcessFuncType combine_process_func_type,
                    K_DType reduce_feature_id) {
  // Step 1: run combine function
  N_DType input_offset = 0;
  N_DType output_offset = 0;
  for (auto i = 0; i < params.num_size; ++i) {
    switch (combine_process_func_type) {
      case kCombineProcessFuncIDFlag:
        IdCombineFunc(params);
        break;
      case kCombineProcessFuncLogIDFlag:
        LogIdCombineFunc(params, input_offset, params.input_nums[i], output_offset, i);
        break;
      case kCombineProcessFuncSumFlag:
        SumCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogSumFlag:
        LogSumCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncMaxFlag:
        MaxCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogMaxFlag:
        LogMaxCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncMinFlag:
        MinCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogMinFlag:
        LogMinCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncAvgFlag:
        AvgCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogAvgFlag:
        LogAvgCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncCosFlag:
        CosCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogCosFlag:
        LogCosCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncDotSumFlag:
        DotSumCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncLogDotSumFlag:
        LogDotSumCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncDotL1NormFlag:
        DotL1NormCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      case kCombineProcessFuncDotL2NormFlag:
        DotL2NormCombineFunc(params, input_offset, params.input_nums[i], output_offset, i, reduce_feature_id);
        break;
      default:
        BLAZE_THROW("unexpected combine func type", combine_process_func_type);
    }
    input_offset += params.input_nums[i];
    output_offset += params.output_nums[i];
  }

  // Step 2: reshape output blob
  std::vector<TIndex> output_id_shape;
  output_id_shape.push_back(static_cast<TIndex>(output_offset));
  params.output_id_blob->Reshape(output_id_shape);
  params.output_value_blob->Reshape(output_id_shape);
}

template <>
bool CombineFuncOp<CPUContext>::RunOnDevice() {
  Blob* id = this->Input(0);
  Blob* value = this->Input(1);
  Blob* num = this->Input(2);

  // check the validity of combine func op
  CheckValid();

  ID_TYPE_SWITCH(id->data_type(), K_DType, {
  TYPE_SWITCH(value->data_type(), V_DType, {
  ID_TYPE_SWITCH(num->data_type(), N_DType, {
    CombineFuncParam<K_DType, V_DType, N_DType> params;
    Setup<K_DType, V_DType, N_DType>(&params);
    RunCombineFunc<K_DType, V_DType, N_DType>(params,
                                              combine_process_func_type_,
                                              static_cast<K_DType>(reduce_feature_id_));
  });
  });
  });

  return true;
}

REGISTER_CPU_OPERATOR(CombineFunc, CombineFuncOp<CPUContext>);
// Input: input_ids, input_values, input_nums.
// Output: output_ids, output_values, output_nums.
OPERATOR_SCHEMA(CombineFunc)
  .NumInputs(3)
  .NumOutputs(3)
  .TypeInferenceFunction([](const OperatorDef& def, const std::vector<DataType>& input_type) {
    ArgumentHelper argument_helper(def);
    std::vector<DataType> ret;
    ret.push_back(input_type[0]);
    ret.push_back(input_type[1]);
    ret.push_back(input_type[2]);
    return ret;
  })
  .SetDoc(R"DOC(
The combine function.
  )DOC")
  .Input(0, "input_ids", "The ids of input feature group")
  .Input(1, "input_values", "The values of input feature group")
  .Input(2, "input_nums", "The id number of each ad in input feature group")
  .Output(0, "output_ids", "The ids of output feature group")
  .Output(1, "output_values", "The values of output feature group")
  .Output(2, "output_nums", "The id number of each ad in output feature group");

}  // namespace blaze
