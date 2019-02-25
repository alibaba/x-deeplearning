/*!
 * \file ulf_importer.cc
 * \brief Convert ulf model into blaze model.
 */
#include "blaze/model_importer/ulf_importer.h"

#include <sstream>
#include <set>

#include "blaze/common/exception.h"
#include "blaze/common/proto_configure.h"
#include "blaze/common/log.h"
#include "blaze/common/string_util.h"
#include "blaze/operator/common_helper.h"

namespace blaze {

ULFImporter::ULFImporter() : ModelImporter() {
  SetProcessNodeFunction("slice_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessSliceLayer(lp);                    
  });
  SetProcessNodeFunction("inner_product_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessInnerProductLayer(lp);                    
  });
  SetProcessNodeFunction("inner_product_layer_ex", [this](const ulf::LayerParameter& lp) {
    return this->ProcessInnerProductLayerEx(lp);
  });
  SetProcessNodeFunction("softmax_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessSoftmaxLayer(lp);                    
  });
  SetProcessNodeFunction("fuse_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessFuseLayer(lp);
  });
  SetProcessNodeFunction("gru_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessGruLayer(lp);
  });
  SetProcessNodeFunction("concat_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessConcatLayer(lp);
  });
  SetProcessNodeFunction("sum_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessSumLayer(lp);
  });
  SetProcessNodeFunction("multiply_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessMultiplyLayer(lp);
  });
  SetProcessNodeFunction("sub_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessSubLayer(lp);
  });
  SetProcessNodeFunction("add_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessAddLayer(lp);
  });
  SetProcessNodeFunction("batchdot_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessBatchDotLayer(lp);
  });
  SetProcessNodeFunction("prelu_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessPreluLayer(lp);
  });
  SetProcessNodeFunction("relu_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessReluLayer(lp);
  });
  SetProcessNodeFunction("sigmoid_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessSigmoidLayer(lp);
  });
  SetProcessNodeFunction("tanh_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessTanhLayer(lp);
  });
  SetProcessNodeFunction("dice_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessDiceLayer(lp);
  });
  SetProcessNodeFunction("bn_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessBnLayer(lp);
  });
  SetProcessNodeFunction("embedding_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessEmbeddingLayer(lp);                       
  });
  SetProcessNodeFunction("div_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessDivLayer(lp);                       
  });
  SetProcessNodeFunction("constant_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessConstantLayer(lp);                       
  });
  SetProcessNodeFunction("reshape_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessReshapeLayer(lp);                       
  });
  SetProcessNodeFunction("broadcast_to_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessBroadcastToLayer(lp);                       
  });
  SetProcessNodeFunction("where_layer", [this](const ulf::LayerParameter& lp) {
    return this->ProcessWhereLayer(lp);
  });
}

void ULFImporter::LoadModel(const char* conf_file, const char* data_file) {
  conf_file_ = conf_file;

  ProtoConfigure config;
  ProtoConfigure::Status rc = config.Init("ulf.NetParameter", conf_file);
  if (rc != ProtoConfigure::kOK) {
    BLAZE_THROW("load model ulf.NetParameter from fila:", conf_file, " failed");
  }
  net_conf_ = *(reinterpret_cast<const ulf::NetParameter*>(config.config()));

  ProtoConfigure param;
  rc = param.Init("ulf.NetWeightsParameter", data_file);
  if (rc != ProtoConfigure::kOK) {
    BLAZE_THROW("load model ulf.NetWeightParameter from file:", data_file, " failed");
  }
  net_param_ = *(reinterpret_cast<const ulf::NetWeightsParameter*>(param.config()));

  if (!Ulf2Blaze()) {
    BLAZE_THROW("ulf2blaze failed");
  }
}

bool ULFImporter::Ulf2Blaze() {
  InitLayerMap();
  // Step1: convert ConstantFill node
  if (!CreateConstantFillNode()) return false;
  // Step3: convert compute node
  return CreateOpNode();
}

void ULFImporter::InitLayerMap() {
  for (size_t k = 0; k < net_param_.layer_weights_params_size(); ++k) {
    const std::string& name = net_param_.layer_weights_params(k).name();
    ulf::LayerWeightsParameter* lwp = net_param_.mutable_layer_weights_params(k);
    layer_param_map_[name] = lwp;
  }
  for (const auto& lp : net_conf_.layer_params()) {
    layer_type_map_[lp.name()] = lp.type();
  }
}

bool ULFImporter::CreateConstantFillNode() {
  for (size_t k = 0; k < net_param_.layer_weights_params_size(); ++k) {
    const auto& name = net_param_.layer_weights_params(k).name();
    const auto& lwp = net_param_.layer_weights_params(k);
    const auto& iter = layer_type_map_.find(name);
    if (iter == layer_type_map_.end()) {
      LOG_ERROR("name=%s is not int net_parameter_conf", name.c_str());
      return false;
    }
    // Now we don't create input's ConstantFillNode
    if (iter->second == "input_layer") continue;

    const auto& op_type = iter->second;
    for (size_t z = 0; z < lwp.blob_datas_size(); ++z) {
      auto op = net_def_.add_op();
      
      std::stringstream ss;
      ss << name << "_" << z;  // name is order by data store sequence.
      op->set_name(ss.str());
      op->set_type("ConstantFill");
      op->add_output(ss.str());
      
      // add arguments, includes: dtype/shape/value
      const auto& blob_data = lwp.blob_datas(z);
      Argument* arg = op->add_arg();
      arg->set_name("dtype");
      arg->set_i(data_type_);
      
      arg = op->add_arg();
      arg->set_name("shape");
      for (size_t j = 0; j < blob_data.shape_size(); ++j) {
        arg->add_ints(blob_data.shape(j));
      }

      arg = op->add_arg();
      arg->set_name("value");
      for (size_t j = 0; j < blob_data.data_size(); ++j) {
        arg->add_floats(blob_data.data(j));
      }
    }
  }
  return true;
}

bool ULFImporter::CreateConstInputConstantFillNode(const ulf::LayerParameter& lp) {
  const auto& name = lp.name();
  const auto& iter = layer_param_map_.find(name);
  if (iter == layer_param_map_.end()) {
    LOG_ERROR("input name: %s not found const params", name.c_str());
    return false;
  }
  if (iter->second->blob_datas_size() != lp.input_param().param_size()) {
    LOG_ERROR("input_param_size=%u blob_datas_size=%u",
              lp.input_param().param_size(), iter->second->blob_datas_size());
    return false;
  }
  for (size_t k = 0; k < lp.input_param().param_size(); ++k) {
    OperatorDef* op = net_def_.add_op();
    
    const std::string& cname = lp.input_param().param(k);
    op->set_name(cname);
    op->set_type("ConstantFill");
    op->add_output(cname);
      
    // add arguments, includes: dtype/shape/value
    const auto& blob_data = iter->second->blob_datas(k);
    Argument* arg = op->add_arg();
    arg->set_name("dtype");
    arg->set_i(data_type_);
      
    arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(1);
    for (size_t j = 0; j < blob_data.shape_size(); ++j) {
      arg->add_ints(blob_data.shape(j));
    }

    arg = op->add_arg();
    arg->set_name("value");
    for (size_t j = 0; j < blob_data.data_size(); ++j) {
      arg->add_floats(blob_data.data(j));
    }
  }
  return true;
}

bool ULFImporter::CreateOpNode() {
  std::set<std::string> external_inputs;
  std::set<std::string> external_outputs;

  const std::string& name = net_conf_.name();
  net_def_.set_name(name);  // set the net name.
  for (const auto& lp : net_conf_.layer_params()) {
    if (lp.type() == "input_layer") {
      // for ulf if the input's top is in InputParam, the top is ConstantOp.
      std::set<std::string> const_params;
      if (lp.has_input_param()) {
        for (const auto& item : lp.input_param().param()) {
          const_params.insert(item);
        }
      }
      for (const auto& top_name : lp.top()) {
        if (!const_params.count(top_name)) {
          external_inputs.insert(top_name);
        }
      }
      if (!const_params.empty()) {
        if (!CreateConstInputConstantFillNode(lp)) return false;
      }
    } else {
      const auto& iter = op_process_func_map_.find(lp.type());
      if (iter == op_process_func_map_.end()) {
        LOG_INFO("layer: %s convertion function not registered", lp.type().c_str());
        return false;
      }
      iter->second(lp);
    }
    // bottom is the input
    for (const auto& bottom_name : lp.bottom()) {
      if (external_outputs.count(bottom_name) != 0) {
        external_outputs.erase(bottom_name);
      }
    }
    // top is the output
    for (const auto& top_name : lp.top()) {
      external_outputs.insert(top_name);
    }
  }
  // set external_input and external_output
  for (auto& item : external_inputs) {
    ValueInfo* value_info = net_def_.add_external_input();
    value_info->set_name(item);

    if (item.length() > strlen(kIdSuffix) &&
        strcmp(item.c_str() + item.length() - strlen(kIdSuffix), kIdSuffix) == 0) {
      value_info->set_input_type(kInputSparseIds);
      value_info->set_feature_name(item.substr(0, item.length() - strlen(kIdSuffix)));
      value_info->set_dtype(kInt64);
      value_info->set_level(sparse_level_[value_info->feature_name()]);
    } else if (item.length() > strlen(kIdNumSuffix) &&
               strcmp(item.c_str() + item.length() - strlen(kIdNumSuffix), kIdNumSuffix) == 0) {
      value_info->set_input_type(kInputSparseSegments);
      value_info->set_feature_name(item.substr(0, item.length() - strlen(kIdNumSuffix)));
      value_info->set_dtype(kInt32);
      value_info->set_level(sparse_level_[value_info->feature_name()]);
    } else if (item.length() > strlen(kValueSuffix) &&
               strcmp(item.c_str() + item.length() - strlen(kValueSuffix), kValueSuffix) == 0) {
      value_info->set_input_type(kInputSparseValues);
      value_info->set_feature_name(item.substr(0, item.length() - strlen(kValueSuffix)));
      value_info->set_dtype(data_type_);
      value_info->set_level(sparse_level_[value_info->feature_name()]);
    } else if (item == kMask) {
      value_info->set_dtype(kInt32);
    } else {
      value_info->set_dtype(data_type_);
    }
  }

  for (auto& item : external_outputs) {
    ValueInfo* value_info = net_def_.add_external_output();
    value_info->set_name(item);
    value_info->set_dtype(data_type_);
  }
  return true;
}

bool ULFImporter::ProcessSliceLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("MultiSlice");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // set param
  Argument* concat_dim_arg = op->add_arg();
  int concat_dim = lp.slice_param().concat_dim();
  concat_dim_arg->set_name("concat_dim");
  concat_dim_arg->set_i(concat_dim);

  Argument* offset_arg = op->add_arg();
  Argument* shape_arg = op->add_arg();
  offset_arg->set_name("offset");
  shape_arg->set_name("shape");
  for (const auto& slice : lp.slice_param().slices()) {
    offset_arg->add_ints(slice.offset());
    for (auto& shape : slice.shape()) {
      shape_arg->add_ints(shape);
    }
  }
  return true;
}

bool ULFImporter::ProcessInnerProductLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Gemm");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }

  std::stringstream ss;
  ss << lp.name() << "_" << 0;
  op->add_input(ss.str());

  bool transA = false;
  bool transB = lp.inner_product_param().transpose();
  if (lp.inner_product_param().bias_term()) {
    // has bias argument
    std::stringstream bias_ss;
    bias_ss << lp.name() << "_" << 1;
    op->add_input(bias_ss.str());
  }

  // set argument
  Argument* arg = op->add_arg();
  arg->set_name("transA");
  arg->set_i(transA);

  arg = op->add_arg();
  arg->set_name("transB");
  arg->set_i(transB);
  return true;
}

bool ULFImporter::ProcessInnerProductLayerEx(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("PrunedGemm");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }

  std::stringstream ss0;
  ss0 << lp.name() << "_" << 0;
  op->add_input(ss0.str());

  std::stringstream ss1;
  ss1 << lp.name() << "_" << 1;
  op->add_input(ss1.str());

  bool transA = false;
  bool transB = lp.inner_product_param().transpose();
  if (lp.inner_product_param().bias_term()) {
    // has bias argument
    std::stringstream bias_ss;
    bias_ss << lp.name() << "_" << 2;
    op->add_input(bias_ss.str());
  }

  // set argument
  Argument* arg = op->add_arg();
  arg->set_name("transA");
  arg->set_i(transA);

  arg = op->add_arg();
  arg->set_name("transB");
  arg->set_i(transB);
  return true;
}

bool ULFImporter::ProcessSoftmaxLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Softmax");

  if (lp.has_softmax_param()) {
    if (lp.softmax_param().has_dim()) {
      Argument* arg = op->add_arg();
      arg->set_name("axis");
      arg->set_i(lp.softmax_param().dim());
    }
  }

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

bool ULFImporter::ProcessFuseLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Fuse");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // get comm_index.
  int comm_index = 0;
  if (lp.has_fuse_param()) {
    comm_index = lp.fuse_param().common_input_index(0);
  }
  Argument* arg = op->add_arg();
  arg->set_name("comm_index");
  arg->set_i(comm_index);
  return true;
}

bool ULFImporter::ProcessGruLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("GRU");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // h2hweight, i2hweight, h2hBias, i2hbias
  const auto& iter = layer_param_map_.find(lp.name());
  if (iter == layer_param_map_.end()) {
    LOG_ERROR("gru layer: %s has not params", lp.name().c_str());
    return false;
  }
  if (iter->second->blob_datas_size() != 4) {
    LOG_ERROR("gru layer param size: %u", iter->second->blob_datas_size());
    return false;
  }
  for (size_t k = 0; k < iter->second->blob_datas_size(); ++k) {
    const ulf::BlobData& blob_datas = iter->second->blob_datas(k);
    std::stringstream ss;
    ss << lp.name() << "_" << k;
    op->add_input(ss.str());
  }

  // Add from deepnet
  auto arg = op->add_arg();
  arg->set_name("from_deepnet");
  arg->set_i(1);

  return true;
}

bool ULFImporter::ProcessConcatLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() < 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Concat");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // set axis
  int axis = 1;
  if (lp.has_concat_param()) {
    axis = lp.concat_param().dim();
  }
  Argument* arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(axis);
  return true;
}

bool ULFImporter::ProcessSumLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("ReduceSum");
  
  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // set axis
  int dim = 1;
  if (lp.has_sum_param()) {
    dim = lp.sum_param().dim(0);
  }
  Argument* arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(dim);
  arg = op->add_arg();
  arg->set_name("keepdims");
  arg->set_i(0);
  return true;
}

bool ULFImporter::ProcessMultiplyLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Mul");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

bool ULFImporter::ProcessSubLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Sub");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

bool ULFImporter::ProcessAddLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Add");

  for (const auto& iname : lp.bottom()) op->add_input(iname);
  for (const auto& oname : lp.top()) op->add_output(oname);
  return true;
}

bool ULFImporter::ProcessBatchDotLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("MatMul");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  bool transA = false, transB = true;
  if (lp.has_batch_dot_param()) {
    transA = lp.batch_dot_param().transpose_a();
    transB = lp.batch_dot_param().transpose_b();
  }
  Argument* arg = op->add_arg();
  arg->set_name("transA");
  arg->set_i(transA);

  arg = op->add_arg();
  arg->set_name("transB");
  arg->set_i(transB);

  arg = op->add_arg();
  arg->set_name("from_deepnet");
  arg->set_i(1);

  return true;
}

bool ULFImporter::ProcessPreluLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("PRelu");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  std::stringstream ss;
  ss << lp.name() << "_" << 0;
  op->add_input(ss.str());
  return true;
}

bool ULFImporter::ProcessReluLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("LeakyReluOp");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  // set alpha to be zero.
  Argument* arg = op->add_arg();
  arg->set_name("alpha");
  arg->set_f(0);
  return true;
}

bool ULFImporter::ProcessSigmoidLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Sigmoid");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

bool ULFImporter::ProcessTanhLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Tanh");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

bool ULFImporter::ProcessDiceLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Dice");

  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  // gamma,mean,var
  const auto& iter = layer_param_map_.find(lp.name());
  if (iter == layer_param_map_.end()) {
    LOG_ERROR("dice layer: %s has not params", lp.name().c_str());
    return false;
  }
  for (size_t k = 0; k < iter->second->blob_datas_size(); ++k) {
    const ulf::BlobData& blob_datas = iter->second->blob_datas(k);
    std::stringstream ss;
    ss << lp.name() << "_" << k;
    op->add_input(ss.str());
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }

  auto arg = op->add_arg();
  arg->set_name("eps");
  if (lp.has_dice_param()) {
    arg->set_f(lp.dice_param().eps());
  } else {
    arg->set_f(1e-8);
  }

  return true;
}

bool ULFImporter::ProcessBnLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 1 || lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }

  OperatorDef* op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("BatchNormalization");

  // x
  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  // beta,gamma,mean,var
  const auto& iter = layer_param_map_.find(lp.name());
  if (iter == layer_param_map_.end()) {
    LOG_ERROR("bn layer: %s has no params", lp.name().c_str());
    return false;
  }
  std::vector<std::string> params;
  for (size_t k = 0; k < iter->second->blob_datas_size(); ++k) {
    const ulf::BlobData& blob_datas = iter->second->blob_datas(k);
    std::stringstream ss;
    ss << lp.name() << "_" << k;
    params.push_back(ss.str());
  }
  std::swap(params[0], params[1]); /// ulf not satisfy onnx standard.
  for (const auto& name : params) {
    op->add_input(name);
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }

  auto arg = op->add_arg();
  arg->set_name("eps");
  if (lp.has_bn_param()) {
    arg->set_f(lp.bn_param().eps());
  } else {
    arg->set_f(0.001);
  }
  return true;
}

bool ULFImporter::ProcessEmbeddingLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() > 0 && lp.bottom_size() % 3 != 0) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Embedding");

  // x
  for (auto k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
    auto splits = Split(lp.bottom(k), kSparseFeatureSep);
    sparse_level_[splits[0]] = lp.embedding_param().level();  // buffer feature level
  }
  // y
  for (auto k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  auto arg = op->add_arg();
  arg->set_name("embedding_config");
  std::string content;
  std::string embedding_config_path = GetParentPath(conf_file_) +
      "/" + lp.embedding_param().embedding_conf_path();
  this->ReadFileContent(embedding_config_path.c_str(), &content);
  arg->set_s(content);

  op->mutable_device_option()->set_device_type(kCPU);
  return true;
}

bool ULFImporter::ProcessDivLayer(const ulf::LayerParameter& lp) {
  if (lp.bottom_size() != 2) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Div");

  // x
  for (auto k = 0; k < lp.bottom_size(); ++k) op->add_input(lp.bottom(k));
  // y
  for (auto k = 0; k < lp.top_size(); ++k) op->add_output(lp.top(k));
  return true;
}

bool ULFImporter::ProcessConstantLayer(const ulf::LayerParameter& lp) {
  if (lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("ConstantFill");

  for (const auto& oname : lp.top()) op->add_output(oname);

  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  for (const auto& dim : lp.constant_param().blob_data().shape()) {
    arg->add_ints(dim);
  }

  arg = op->add_arg();
  arg->set_name("value");
  for (const auto& data : lp.constant_param().blob_data().data()) {
    arg->add_floats(data);
  }
  return true;
}

bool ULFImporter::ProcessReshapeLayer(const ulf::LayerParameter& lp) {
  if (lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto tmp_op = net_def_.add_op();
  tmp_op->set_name(lp.name() + "_reshape");
  tmp_op->set_type("ConstantFill");
  tmp_op->add_output(tmp_op->name());

  auto arg = tmp_op->add_arg();
  arg->set_name("dtype");
  arg->set_i(kInt32);

  arg = tmp_op->add_arg();
  arg->set_name("shape");
  arg->add_ints(lp.reshape_param().shape_size());

  arg = tmp_op->add_arg();
  arg->set_name("value");
  for (const auto& dim : lp.reshape_param().shape()) {
    arg->add_ints(dim);
  }
  tmp_op->mutable_device_option()->set_device_type(kCPU);

  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Reshape");
  for (const auto& oname : lp.top()) op->add_output(oname);
  for (const auto& iname : lp.bottom()) op->add_input(iname);
  op->add_input(tmp_op->output(0));
  return true;
}

bool ULFImporter::ProcessBroadcastToLayer(const ulf::LayerParameter& lp) {
  if (lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("BroadcastTo");
  for (const auto& oname : lp.top()) op->add_output(oname);
  for (const auto& iname : lp.bottom()) op->add_input(iname);

  if (lp.has_broadcast_to_param()) {
    auto arg = op->add_arg();
    arg->set_name("shape");
    for (const auto& shape : lp.broadcast_to_param().shape()) {
      arg->add_ints(shape);
    }
  }
  return true;
}

bool ULFImporter::ProcessWhereLayer(const ulf::LayerParameter& lp) {
  if (lp.top_size() != 1) {
    LOG_ERROR("Argument config failed: %s", lp.DebugString().c_str());
    return false;
  }
  auto op = net_def_.add_op();
  op->set_name(lp.name());
  op->set_type("Where");
  for (size_t k = 0; k < lp.bottom_size(); ++k) {
    op->add_input(lp.bottom(k));
  }
  for (size_t k = 0; k < lp.top_size(); ++k) {
    op->add_output(lp.top(k));
  }
  return true;
}

void ULFImporter::SetProcessNodeFunction(const std::string& name, ULFImporter::ProcessOpNodeFunction function) {
  op_process_func_map_[name] = function;
}

}  // namespace blaze

