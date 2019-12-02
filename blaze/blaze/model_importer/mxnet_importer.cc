/*
 * \file mxnet_importer.cc
 * \brief The mxnet importer
 */
#include "blaze/model_importer/mxnet_importer.h"

#include <fstream>
#include <sstream>

#include "blaze/common/log.h"
#include "blaze/common/string_util.h"

namespace blaze {

BLAZE_JSON_ENABLE_ANY(int, int);
BLAZE_JSON_ENABLE_ANY(size_t, size_t);
BLAZE_JSON_ENABLE_ANY(std::string, str);
BLAZE_JSON_ENABLE_ANY(std::vector<int>, list_int);
BLAZE_JSON_ENABLE_ANY(std::vector<std::string>, list_str);

MXNetImporter::MXNetImporter() {
  SetProcessNodeFunction(
      "null", [this](const JSONNode& node) { this->ProcessNullOp(node); });
  SetProcessNodeFunction(
      "Concat", [this](const JSONNode& node) { this->ProcessConcatOp(node); });
  SetProcessNodeFunction(
      "FullyConnected", [this](const JSONNode& node) { this->ProcessFullyConnectedOp(node); });
  SetProcessNodeFunction(
      "BatchNorm", [this](const JSONNode& node) { this->ProcessBatchNormOp(node); });
  SetProcessNodeFunction(
      "Activation", [this](const JSONNode& node) { this->ProcessActivationOp(node); });
  SetProcessNodeFunction(
      "_rminus_scalar", [this](const JSONNode& node) { this->ProcessRMimusScalarOp(node); });
  SetProcessNodeFunction(
      "elemwise_mul", [this](const JSONNode& node) { this->ProcessElemwiseMulOp(node); });
  SetProcessNodeFunction(
      "elemwise_div", [this](const JSONNode& node) { this->ProcessElemwiseDivOp(node); });
  SetProcessNodeFunction(
      "broadcast_mul", [this](const JSONNode& node) { this->ProcessBroadcastMulOp(node); });
  SetProcessNodeFunction(
      "elemwise_add", [this](const JSONNode& node) { this->ProcessElemwiseAddOp(node); });
  SetProcessNodeFunction(
      "SoftmaxOutput", [this](const JSONNode& node) { this->ProcessSoftmaxOutputOp(node); });
  SetProcessNodeFunction(
      "take", [this](const JSONNode& node) { this->ProcessTakeOp(node); });
  SetProcessNodeFunction(
      "slice_axis", [this](const JSONNode& node) { this->ProcessSliceAxisOp(node); });
  SetProcessNodeFunction(
      "sum", [this](const JSONNode& node) { this->ProcessSumOp(node); });
  SetProcessNodeFunction(
      "Reshape", [this](const JSONNode& node) { this->ProcessReshapeOp(node); });
  SetProcessNodeFunction(
      "batch_dot", [this](const JSONNode& node) { this->ProcessBatchdotOp(node); });
  SetProcessNodeFunction(
      "SoftmaxActivation", [this](const JSONNode& node) { this->ProcessSoftmaxActivationOp(node); });
  SetProcessNodeFunction(
      "_plus_scalar", [this](const JSONNode& node) { this->ProcessPlusScalarOp(node); });
  SetProcessNodeFunction(
      "SliceChannel", [this](const JSONNode& node) { this->ProcessSliceChannelOp(node); });
  SetProcessNodeFunction(
      "_zeros", [this](const JSONNode& node) { this->ProcessZerosOp(node); });
  SetProcessNodeFunction(
      "_equal_scalar", [this](const JSONNode& node) { this->ProcessEqualScalarOp(node); });
  SetProcessNodeFunction(
      "_not_equal_scalar", [this](const JSONNode& node) { this->ProcessNotEqualScalarOp(node); });
  SetProcessNodeFunction(
      "LeakyReLU", [this](const JSONNode& node) { this->ProcessLeakyReLUOp(node); });
  SetProcessNodeFunction(
      "_div_scalar", [this](const JSONNode& node) { this->ProcessDivScalarOp(node); });
  SetProcessNodeFunction(
      "_rdiv_scalar", [this](const JSONNode& node) { this->ProcessRDivScalarOp(node); });
  SetProcessNodeFunction(
      "_minus_scalar", [this](const JSONNode& node) { this->ProcessMinusScalarOp(node); });
  SetProcessNodeFunction(
      "_mul_scalar", [this](const JSONNode& node) { this->ProcessMulScalarOp(node); });
  SetProcessNodeFunction(
      "_power_scalar", [this](const JSONNode& node) { this->ProcessPowerScalarOp(node); });
  SetProcessNodeFunction(
      "broadcast_to", [this](const JSONNode& node) { this->ProcessBroadcastToOp(node); });
  SetProcessNodeFunction(
      "_ones", [this](const JSONNode& node) { this->ProcessOnesOp(node); });
  SetProcessNodeFunction(
      "broadcast_add", [this](const JSONNode& node) { this->ProcessBroadcastAddOp(node); });
  SetProcessNodeFunction(
      "add_n", [this](const JSONNode& node) { this->ProcessAddNOp(node); });
}

void MXNetImporter::LoadModel(const char* conf_file, const char* data_file) {
  std::ifstream in(conf_file, std::ifstream::in);
  LoadModel(in, data_file);
}

void MXNetImporter::LoadModel(std::istream& is, const char* data_file) {
  // Step1: load json.
  JSONReader reader(&is);
  jgraph_.Load(&reader);
  
  // Step2: load ndarray map
  FileStream stream(data_file, true);
  mparam_.Load(&stream);
  
  // Step3: Convert mxnet model to blaze
  MXNet2Blaze();
}

void MXNetImporter::MXNet2Blaze() {
  CreateConstantFillNode();
  CreateOpNode();

  std::unordered_set<std::string> oname_set, external_input, external_output;
  for (const auto& op : net_def_.op()) {
    for (const auto& iname : op.input()) {
      if (!oname_set.count(iname)) {
        external_input.insert(iname);
      }
      external_output.erase(iname);
    }
    for (const auto& oname : op.output()) {
      oname_set.insert(oname);
      if (op.type() != "ConstantFill") {
        external_output.insert(oname);
      }
    }
  }

  for (const auto& name : external_input) {
    auto value_info = net_def_.add_external_input();
    value_info->set_name(name);
    value_info->set_dtype(data_type_);
  }
  for (const auto& name : external_output) {
    auto value_info = net_def_.add_external_output();
    value_info->set_name(name);
    value_info->set_dtype(data_type_);
  }
  //LOG_INFO("net_def=%s", net_def_.DebugString().c_str());
}

void MXNetImporter::CreateConstantFillNode() {
  for (auto i = 0; i < mparam_.keys.size(); ++i) {
    const auto& name = mparam_.keys[i];
    const auto& ndarray = mparam_.ndarray[i];
    LOG_DEBUG("mx param name=%s", name.c_str());

    // Add a ConstantOp
    auto op = net_def_.add_op();
    op->set_name(name);
    op->set_type("ConstantFill");
    op->add_output(name);

    // add argument, includes: dtype/shape/value
    auto arg = op->add_arg();
    arg->set_name("dtype");
    arg->set_i(ndarray.data_type);

    arg = op->add_arg();
    arg->set_name("shape");
    for (const auto& dim : ndarray.shape) {
      arg->add_ints(dim);
    }

    arg = op->add_arg();
    arg->set_name("value");
    for (auto i = 0; i < ndarray.data.size(); ++i) {
      if (IsIntegerType(ndarray.data_type)) {
        arg->add_ints(ndarray.data[i].i);
      } else {
        arg->add_floats(ndarray.data[i].f);
      }
    }
  }
}

void MXNetImporter::CreateOpNode() {
  static std::unordered_set<std::string> black_list_op = {
    "BlockGrad",
    "log",
  };
  for (auto i = 0; i < jgraph_.nodes.size(); ++i) {
    const auto& node = jgraph_.nodes[i];
    const auto& op_type = node.op_type;
    if (black_list_op.count(op_type)) continue;
    const auto& iter = op_process_func_map_.find(op_type);
    CHECK_TRUE(iter != op_process_func_map_.end(), " parse op_type=", op_type, " is not registered");
    iter->second(node);
  }
}

void MXNetImporter::ProcessNullOp(const JSONNode& node) {
}

void MXNetImporter::ProcessConcatOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Concat");
  
  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(GetIntegerAttr(node, "dim", 1));
}

void MXNetImporter::ProcessFullyConnectedOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Gemm");

  auto arg = op->add_arg();
  arg->set_name("transB");
  arg->set_i(1);
}

void MXNetImporter::ProcessBatchNormOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "BatchNormalization");

  auto arg = op->add_arg();
  arg->set_name("eps");
  arg->set_f(GetFloatAttr(node, "eps", 1e-3f));
}

void MXNetImporter::ProcessActivationOp(const JSONNode& node) {
  auto act_type = GetStrAttr(node, "act_type", "sigmoid");

  if (!strcmp(act_type, "sigmoid")) AddOperatorDef(node, "Sigmoid");
  else if (!strcmp(act_type, "relu")) {
    auto op = AddOperatorDef(node, "LeakyRelu");
    auto arg = op->add_arg();
    arg->set_name("alpha");
    arg->set_f(0);
  }
  else if (!strcmp(act_type, "tanh")) AddOperatorDef(node, "Tanh");
  else { BLAZE_THROW("act_type=", act_type, " not support"); }
}

void MXNetImporter::ProcessRMimusScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);
  
  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_rminus_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Sub");
  op->add_input(scalar_oname);

  *(op->mutable_input(1)) = op->input(0);
  *(op->mutable_input(0)) = scalar_oname;
}

void MXNetImporter::ProcessElemwiseMulOp(const JSONNode& node) {
  AddOperatorDef(node, "Mul");
}

void MXNetImporter::ProcessBroadcastMulOp(const JSONNode& node) {
  AddOperatorDef(node, "Mul");
}

void MXNetImporter::ProcessElemwiseDivOp(const JSONNode& node) {
  AddOperatorDef(node, "Div");
}

void MXNetImporter::ProcessElemwiseAddOp(const JSONNode& node) {
  AddOperatorDef(node, "Add");
}

void MXNetImporter::ProcessSoftmaxOutputOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Softmax");

  auto iname = op->input(0);
  op->clear_input();
  op->add_input(iname);

  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
}

void MXNetImporter::ProcessSoftmaxActivationOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Softmax");
  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
}

void MXNetImporter::ProcessTakeOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Gather");

  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(GetIntegerAttr(node, "dim", 0));
}

void MXNetImporter::ProcessSliceAxisOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "Slice");

  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(GetIntegerAttr(node, "axis", 1));

  arg = op->add_arg();
  arg->set_name("start");
  arg->set_i(GetIntegerAttr(node, "begin", 0));

  arg = op->add_arg();
  arg->set_name("end");
  arg->set_i(GetIntegerAttr(node, "end", 1));
}

void MXNetImporter::ProcessSumOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "ReduceSum");

  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(GetIntegerAttr(node, "axis", 1));

  arg = op->add_arg();
  arg->set_name("keepdims");
  auto keepdims = GetStrAttr(node, "keepdims", "False");
  if (strcmp(keepdims, "True") == 0) {  // if keepdims=True
    arg->set_i(1);
  } else {
    arg->set_i(0);
  }
}

void MXNetImporter::ProcessReshapeOp(const JSONNode& node) {
  // Add Constant Op.
  auto shape = GetStrAttr(node, "shape", "(0)");
  std::string trim_shape = std::string(shape).substr(1, strlen(shape) - 2);
  auto segment = Split(trim_shape, ", ");

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_reshape");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(kInt32);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(segment.size());

  arg = op->add_arg();
  arg->set_name("value");
  for (auto i = 0; i < segment.size(); ++i) {
    arg->add_ints(std::stoi(segment[i].c_str()));
  }
  const std::string& oname = op->output(0);

  op->mutable_device_option()->set_device_type(kCPU);

  // Create Reshape node
  op = AddOperatorDef(node, "Reshape");
  op->add_input(oname);
}

void MXNetImporter::ProcessBatchdotOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "MatMul");

  auto transpose_b = GetStrAttr(node, "transpose_b", "True");
  auto arg = op->add_arg();
  arg->set_name("transB");
  arg->set_i(strcmp(transpose_b, "True") == 0 ? 1 : 0);
}

void MXNetImporter::ProcessPlusScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1e-7);
  
  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_plus_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Add");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessSliceChannelOp(const JSONNode& node) {
  auto axis = GetIntegerAttr(node, "axis", 1);
  auto num_output = GetIntegerAttr(node, "num_outputs", 1);

  auto op = AddOperatorDef(node, "Split", num_output);
  auto arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(axis);
}

void MXNetImporter::ProcessZerosOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "ConstantFill");

  auto dtype_str = GetStrAttr(node, "dtype", "float32");
  auto shape_str = GetStrAttr(node, "shape", "(0, 0)");

  auto dtype = Str2DataType(dtype_str);
  std::string trim_shape = std::string(shape_str).substr(1, strlen(shape_str) - 2);
  auto segment = Split(trim_shape, ", ");
  size_t zeros_size = 1;

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  for (const auto& item : segment) {
    size_t dim = std::stoi(item.c_str()) <= 0 ? 1 : std::stoi(item.c_str());
    arg->add_ints(dim);
    zeros_size *= dim;
  }

  arg = op->add_arg();
  arg->set_name("value");
  for (auto i = 0; i < zeros_size; ++i) {
    arg->add_floats(0);
  }
}

void MXNetImporter::ProcessEqualScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 0.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_equal_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Equal");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessNotEqualScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 0.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_notequal_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "NotEqual");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessLeakyReLUOp(const JSONNode& node) {
  AddOperatorDef(node, "PRelu");
}

void MXNetImporter::ProcessDivScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_div_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Div");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessRDivScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_rdiv_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Div");
  op->add_input(scalar_oname);

  *(op->mutable_input(1)) = op->input(0);
  *(op->mutable_input(0)) = scalar_oname;
}

void MXNetImporter::ProcessMinusScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_minus_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Sub");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessMulScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_mul_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Mul");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessPowerScalarOp(const JSONNode& node) {
  auto scalar = GetFloatAttr(node, "scalar", 1.0);

  // add constant fill op.
  const auto& op_name = node.attrs.name;
  auto op = net_def_.add_op();
  op->set_name(op_name + "_pow_scalar");
  op->set_type("ConstantFill");
  op->add_output(op->name());

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);

  arg = op->add_arg();
  arg->set_name("value");
  arg->add_floats(scalar);

  const auto& scalar_oname = op->output(0);
  op = AddOperatorDef(node, "Pow");
  op->add_input(scalar_oname);
}

void MXNetImporter::ProcessBroadcastToOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "BroadcastTo", 1);

  auto shape = GetStrAttr(node, "shape", "(0)");
  std::string trim_shape = std::string(shape).substr(1, strlen(shape) - 2);
  auto segment = Split(trim_shape, ", ");
  
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (const auto& seg : segment) {
    arg->add_ints(std::stoi(seg.c_str()));
  }
}

void MXNetImporter::ProcessOnesOp(const JSONNode& node) {
  auto op = AddOperatorDef(node, "ConstantFill");

  auto dtype_str = GetStrAttr(node, "dtype", "float32");
  auto shape_str = GetStrAttr(node, "shape", "(0, 0)");

  auto dtype = Str2DataType(dtype_str);
  std::string trim_shape = std::string(shape_str).substr(1, strlen(shape_str) - 2);
  auto segment = Split(trim_shape, ", ");
  size_t ones_size = 1;

  // add argument, includes: dtype/shape/value
  auto arg = op->add_arg();
  arg->set_name("dtype");
  arg->set_i(data_type_);

  arg = op->add_arg();
  arg->set_name("shape");
  for (const auto& item : segment) {
    size_t dim = std::stoi(item.c_str()) <= 0 ? 1 : std::stoi(item.c_str());
    arg->add_ints(dim);
    ones_size *= dim;
  }

  arg = op->add_arg();
  arg->set_name("value");
  for (auto i = 0; i < ones_size; ++i) {
    arg->add_floats(1.0);
  }
}

void MXNetImporter::ProcessBroadcastAddOp(const JSONNode& node) {
  AddOperatorDef(node, "Add", 1);
}

void MXNetImporter::ProcessAddNOp(const JSONNode& node) {
  auto num_args = GetIntegerAttr(node, "num_args", 2);
  if (num_args == 2) {
    AddOperatorDef(node, "Add", 1);
  } else {
    BLAZE_THROW("Not supported add_n. num_args=", num_args);
  }
}

OperatorDef* MXNetImporter::AddOperatorDef(const JSONNode& node, const char* op_type, int onum) {
  const auto& op_name = node.attrs.name;

  auto op = net_def_.add_op();
  op->set_name(op_name);
  op->set_type(op_type);
  // init iname
  for (const auto& entry : node.inputs) {
    auto node_id = entry.node_id;
    auto index = entry.index;
    if (jgraph_.nodes[node_id].op_type == "null" || index == 0) {
      auto iname = jgraph_.nodes[node_id].attrs.name;
      op->add_input(iname);
    } else {
      auto iname = jgraph_.nodes[node_id].attrs.name + ":" + std::to_string(index);
      op->add_input(iname);
    }
  }
  // init oname
  for (auto i = 0; i < onum; ++i) {
    if (i == 0) {
      op->add_output(op_name);
    } else {
      op->add_output(op_name + ":" + std::to_string(i));
    }
  }
  return op;
}

float MXNetImporter::GetFloatAttr(const JSONNode& node, const char* key, float default_val) {
  const auto& iter = node.attrs.dict.find(key);
  if (iter == node.attrs.dict.end()) return default_val;
  return std::stof(iter->second.c_str());
}

int64_t MXNetImporter::GetIntegerAttr(const JSONNode& node, const char* key, int64_t default_val) {
  const auto& iter = node.attrs.dict.find(key);
  if (iter == node.attrs.dict.end()) return default_val;
  return std::stol(iter->second.c_str());
}

const char* MXNetImporter::GetStrAttr(const JSONNode& node, const char* key, const char* default_val) {
  const auto& iter = node.attrs.dict.find(key);
  if (iter == node.attrs.dict.end()) return default_val;
  return iter->second.c_str();
}

void MXNetImporter::SetProcessNodeFunction(
    const std::string& name, MXNetImporter::ProcessOpNodeFunction function) {
  op_process_func_map_[name] = function;
}

DataType MXNetImporter::Str2DataType(const char* data_type_str) {
  if (!strcmp(data_type_str, "float32")) {
    return kFloat;
  } else if (!strcmp(data_type_str, "float64")) {
    return kDouble;
  } else {
    BLAZE_THROW("unkown datatype %s for mxnet datatype conversion", data_type_str);
    return kFloat;
  }
}

}  // namespace blaze 
