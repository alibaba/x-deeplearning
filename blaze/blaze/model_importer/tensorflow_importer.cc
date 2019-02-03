/*
 * \file tensorflow_importer.cc
 * \brief The tensorflow importer
 */
#include "blaze/model_importer/tensorflow_importer.h"

#include "blaze/common/proto_configure.h"
#include "blaze/common/string_util.h"

namespace blaze {

TensorFlowImporter::TensorFlowImporter() {
  SetProcessNodeFunction("Placeholder", [this](const tensorflow::NodeDef& node) {
                         this->ProcessPlaceholderOp(node); });
  SetProcessNodeFunction("Const", [this](const tensorflow::NodeDef& node) {
                         this->ProcessConstOp(node); });
  SetProcessNodeFunction("Identity", [this](const tensorflow::NodeDef& node) {
                         this->ProcessIdentityOp(node); });
  SetProcessNodeFunction("MatMul", [this](const tensorflow::NodeDef& node) {
                         this->ProcessMatMulOp(node); });
  SetProcessNodeFunction("Add", [this](const tensorflow::NodeDef& node) {
                         this->ProcessAddOp(node); });
  SetProcessNodeFunction("Mul", [this](const tensorflow::NodeDef& node) {
                         this->ProcessMulOp(node); });
  SetProcessNodeFunction("Maximum", [this](const tensorflow::NodeDef& node) {
                         this->ProcessMaximumOp(node); });
  SetProcessNodeFunction("Softmax", [this](const tensorflow::NodeDef& node) {
                         this->ProcessSoftmaxOp(node); });
}

void TensorFlowImporter::LoadModel(const char* conf_file, const char* data_file) {
  ProtoConfigure config;
  auto rc = config.Init("tensorflow.MetaGraphDef", conf_file);
  if (rc != ProtoConfigure::kOK) {
    BLAZE_THROW("Parse tensorflow.MetaGraphDef failed file=", conf_file);
  }
  graph_def_ = (*(reinterpret_cast<const tensorflow::MetaGraphDef*>(config.config()))).graph_def();
  LoadCkpt(data_file);

  Tensorflow2Blaze();
}

void TensorFlowImporter::LoadModelFromString(const std::string& conf_str, const char* data_file) {
  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.ParseFromArray(const_cast<char*>(conf_str.c_str()), conf_str.size());
  graph_def_ = meta_graph_def.graph_def();
  LOG_INFO("graph_def_=%s", graph_def_.DebugString().c_str());
  
  LoadCkpt(data_file);

  Tensorflow2Blaze();
}

void TensorFlowImporter::LoadCkpt(const char* data_file) {
  tf_param_.Load(data_file);
  for (auto i = 0; i < tf_param_.keys.size(); ++i) {
    LOG_INFO("name=%s", tf_param_.keys[i].c_str());
    const auto& array = tf_param_.ndarray[i];
    size_t size = 1;
    for (const auto& dim : array.shape) size *= dim;
    for (auto i = 0; i < size; ++i) {
      LOG_DEBUG("data[%d]=%f", i, array.data[i].f);
    }
  }
  LOG_ERROR("Loading Chpt is not implemented");
}

void TensorFlowImporter::Tensorflow2Blaze() {
  // Step1: init name_rewrite
#if 0
  for (const auto& node : graph_def_.node()) {
    if (node.op() == "Identity") {
      std::string input_node_name;
      int index;
      GetInput(node.input(0), &input_node_name, &index);
      if (const_name_set_.count(input_node_name)) {
        name_rewrite_[node.name()] = input_node_name;
      }
    } else if (node.op() == "Const") {
      const_name_set_.insert(node.name());
    }
  }

  // Step2: op conversion
  for (const auto& node : graph_def_.node()) {
    const auto& op_type = node.op();
    const auto& iter = op_process_func_map_.find(op_type);
    CHECK_TRUE(iter != op_process_func_map_.end(),
               " parse op_type=", op_type, " is not registered");
    iter->second(node);
  }
#endif
  LOG_INFO("net_def=%s", net_def_.DebugString().c_str());
}

void TensorFlowImporter::ProcessPlaceholderOp(const tensorflow::NodeDef& node) {
  auto value_info = net_def_.add_external_input();
  value_info->set_name(node.name());
  
  // set dtype
  const auto& dtype_iter = node.attr().find("dtype");
  CHECK_TRUE(dtype_iter != node.attr().end());
  value_info->set_dtype(GetDataType(dtype_iter->second.type()));

  // set shape
}

void TensorFlowImporter::ProcessConstOp(const tensorflow::NodeDef& node) {
  auto op = AddOperatorDef(node, "ConstantFill");
}

void TensorFlowImporter::ProcessIdentityOp(const tensorflow::NodeDef& node) {
  const auto& input = node.input(0);
  std::string iname;
  int index;
  GetInput(input, &iname, &index);
  if (const_name_set_.count(iname)) {
    return;
  }
  AddOperatorDef(node, "Identity");
}

void TensorFlowImporter::ProcessMatMulOp(const tensorflow::NodeDef& node) {
  const auto& transpose_a_iter = node.attr().find("transpose_a");
  const auto& transpose_b_iter = node.attr().find("transpose_b");
  CHECK_NE(transpose_a_iter, node.attr().end());
  CHECK_NE(transpose_b_iter, node.attr().end());

  auto op = AddOperatorDef(node, "Gemm");
  auto arg = op->add_arg();
  arg->set_name("transepose_a");
  arg->set_i(transpose_a_iter->second.b() ? 1 : 0);

  arg = op->add_arg();
  arg->set_name("transpose_b");
  arg->set_i(transpose_b_iter->second.b() ? 1 : 0);
}

void TensorFlowImporter::ProcessAddOp(const tensorflow::NodeDef& node) {
  AddOperatorDef(node, "Add");
}

void TensorFlowImporter::ProcessMulOp(const tensorflow::NodeDef& node) {
  AddOperatorDef(node, "Mul");
}

void TensorFlowImporter::ProcessMaximumOp(const tensorflow::NodeDef& node) {
  AddOperatorDef(node, "Max");
}

void TensorFlowImporter::ProcessSoftmaxOp(const tensorflow::NodeDef& node) {
  AddOperatorDef(node, "Softmax");
}

OperatorDef* TensorFlowImporter::AddOperatorDef(
    const tensorflow::NodeDef& node, const char* op_type, int onum) {
  const auto& op_name = node.name();

  auto op = net_def_.add_op();
  op->set_name(op_name);
  op->set_type(op_type);
  
  // init iname
  std::string input_node_name;
  int index;
  for (const auto& entry : node.input()) {
    GetInput(entry, &input_node_name, &index);
    const auto& name_rewrite_iter = name_rewrite_.find(input_node_name);
    if (name_rewrite_iter != name_rewrite_.end()) {
      input_node_name = name_rewrite_iter->second;
    }
    op->add_input(input_node_name + (index == 0 ? "" : std::to_string(index)));
  }
  
  // init oname
  for (auto i = 0; i < onum; ++i) {
    op->add_output(op_name + (i == 0 ? "" : std::to_string(i)));
  }
  return op;
}

void TensorFlowImporter::GetInput(const std::string& input_str, std::string* node_name, int* index) {
  auto splits = Split(input_str, ':');
  CHECK_TRUE(splits.size() >= 1);
  *node_name = splits[0];
  *index = 0;
  if (splits.size() > 1) {
    *index = std::stoi(splits[1].c_str());
  }
}

DataType TensorFlowImporter::GetDataType(tensorflow::DataType data_type) {
  switch (data_type) {
    case tensorflow::DT_FLOAT:
      return kFloat;
    case tensorflow::DT_DOUBLE:
      return kDouble;
    case tensorflow::DT_INT32:
      return kInt32;
    default:
      BLAZE_THROW("tensoflow data_type=", data_type, " is not supported");
  }
}

void TensorFlowImporter::SetProcessNodeFunction(
    const std::string& name, TensorFlowImporter::ProcessOpNodeFunction function) {
  op_process_func_map_[name] = function;
}

}  // namespace blaze
