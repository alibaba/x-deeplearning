/*
 * \file store_pull_op.cc
 * \brief The embedding operation, pull sparse parameter from local storage or
 * distributed storage.
 */
#include "blaze/operator/sparse_op/embedding/embedding_op.h"
#include "blaze/common/proto_configure.h"

namespace blaze {

template <>
EmbeddingOp<CPUContext>::EmbeddingOp(const OperatorDef& def, Workspace* workspace) :
    Operator<CPUContext>(def, workspace) {
  // Get sparse puller    
  sparse_puller_ = workspace->GetSparsePuller();

  // each feature group has it's own udf type.
  std::string embedding_config_str = OperatorBase::GetSingleArgument<std::string>("embedding_config", "");
  try {
    ProtoConfigure proto_conf;
    proto_conf.InitByTextConf("blaze.EmbeddingConfig", embedding_config_str);
    const EmbeddingConfig* embedding_config = dynamic_cast<const EmbeddingConfig*>(proto_conf.config());
    BLAZE_CONDITION_THROW(embedding_config != nullptr,
                          "Parse embedding config failed: ",
                          embedding_config_str);
    Init(*embedding_config);
  } catch (...) {
    BLAZE_THROW("Parse embedding config failed:", embedding_config_str);
  }

  // init input fg order
  BLAZE_CONDITION_THROW(def.input_size() % 3 == 0,
                        "input size is not multiply of 3");
  auto input_size = def.input_size();
  for (auto i = 0, p = 0; i < input_size; i += 3, ++p) {
    const std::string& id_input_name = this->operator_def().input(i);
    const std::string& value_input_name = this->operator_def().input(i + 1);
    const std::string& num_input_name = this->operator_def().input(i + 2);
    std::string fg_name = id_input_name.substr(0, id_input_name.rfind(kIdSuffix));
    std::string value_fg_name = value_input_name.substr(0, value_input_name.rfind(kValueSuffix));
    std::string num_fg_name = num_input_name.substr(0, num_input_name.rfind(kIdNumSuffix));
    BLAZE_CONDITION_THROW(!fg_name.empty(),
                          "invalid input name=",
                          id_input_name);
    BLAZE_CONDITION_THROW(!value_fg_name.empty(),
                          "invalid input name=",
                          value_input_name);
    BLAZE_CONDITION_THROW(!num_fg_name.empty(),
                          "invalid input name=",
                          num_fg_name);
    BLAZE_CONDITION_THROW(fg_name == value_fg_name && fg_name == num_fg_name,
                          "id input name=",
                          id_input_name,
                          " value_input_name=",
                          value_input_name,
                          " num_input_name=",
                          num_input_name);
    input_fg_idx_map_[fg_name] = p;
  }
}

template <>
bool EmbeddingOp<CPUContext>::RunOnDevice() {
  Blob* id = this->Input(0);
  Blob* value = this->Input(1);
  Blob* id_num = this->Input(2);

  // check the validity of embedding op
  CheckValid();

  ID_TYPE_SWITCH(id->data_type(), K_DType, {
  TYPE_SWITCH(value->data_type(), V_DType, {
  ID_TYPE_SWITCH(id_num->data_type(), N_DType, {
    EmbeddingParam<K_DType, V_DType, N_DType> params;
    Setup<K_DType, V_DType, N_DType>(&params);
    RunEmbedding<K_DType, V_DType, N_DType>(params,
                                            id->data_type(),
                                            value->data_type(),
                                            id_num->data_type());
  });
  });
  });

  return true;
}

REGISTER_CPU_OPERATOR(Embedding, EmbeddingOp<CPUContext>);

// Input: id, value, id_num
// Output: dense embedding blocks
OPERATOR_SCHEMA(Embedding)
  .NumInputs(3, INT_MAX)
  .NumOutputs(1, INT_MAX)
  .TypeInferenceFunction([](const OperatorDef& def, const std::vector<DataType>& input_type) {
    std::vector<DataType> ret;
    ret.resize(def.output_size(), input_type[1]);
    return ret;
  })
  .SetDoc(R"DOC(
The embedding op.
  )DOC");

}  // namespace blaze
