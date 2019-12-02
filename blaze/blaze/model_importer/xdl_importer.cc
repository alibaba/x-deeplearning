/*
 * \file xdl_importer.cc
 * \brief The xdl importer. 
 */
#include "blaze/model_importer/xdl_importer.h"
#include "blaze/common/proto_configure.h"
#include "blaze/common/log.h"
#include "blaze/common/exception.h"
#include "blaze/common/string_util.h"
#include "blaze/operator/common_helper.h"
#include "blaze/optimizer/optimizer.h"

namespace {
const char kDelimiter = '.';
const int kUniqueOpOutputSize = 4;
const int kPsSparsePullOpOutputSize = 1;
const int kPsPullOpOutputSize = 1;
const int kConstantOutputSize = 1;
const int kKSumOutputSize = 1;
const int kTileOutputSize = 1;
const int kTakeOpOutputSize = 1;
const char kVarNameDelim = ',';
}  // namespace

namespace blaze {

XdlImporter::XdlImporter() : ModelImporter() {
  net_def_.set_run_mode("hybrid");

  SetProcessNodeFunction("GetBatch", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessGetBatchOp(nd);
  });
  SetProcessNodeFunction("PsIsInitializedOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsIsInitializedOp(nd);
  });
  SetProcessNodeFunction("PsNormalInitializerOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsNormalInitializerOp(nd);
  });
  SetProcessNodeFunction("PsRegisterVariableOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsRegisterVariableOp(nd);
  });
  SetProcessNodeFunction("PsConstantInitializerOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsConstantInitializerOp(nd);
  });
  SetProcessNodeFunction("PsTruncatedNormalInitializerOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsTruncatedNormalInitializerOp(nd);
  });
  SetProcessNodeFunction("PsIdentityInitializerOp", [this](const xdl::proto::NodeDef &nd) {
    this->ProcessPsIdentityInitializerOp(nd);
  });
  SetProcessNodeFunction("Unique", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessUniqueOp(nd);
  });
  SetProcessNodeFunction("TakeOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessTakeOp(nd);
  });
  SetProcessNodeFunction("PsSparsePullOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsSparsePullOp(nd);
  });
  SetProcessNodeFunction("PsPullOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessPsPullOp(nd);
  });
  SetProcessNodeFunction("_Constant", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessConstantOp(nd);
  });
  SetProcessNodeFunction("KSum", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessKSumOp(nd);
  });
  SetProcessNodeFunction("Tile", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessTileOp(nd);
  });
  SetProcessNodeFunction("MxnetBackendOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessMxnetBackendOp(nd);
  });
  SetProcessNodeFunction("TFBackendOp", [this](const xdl::proto::NodeDef& nd) {
    this->ProcessTFBackendOp(nd);                       
  });
}

void XdlImporter::LoadModel(const char *graph_def_conf_file, const char* data_file) {
  data_file_ = data_file;

  ProtoConfigure config;
  ProtoConfigure::Status rc = config.Init("xdl.proto.GraphDef", graph_def_conf_file);
  if (rc != ProtoConfigure::kOK) {
    BLAZE_THROW("load model xdl.proto.GraphDef from file=", graph_def_conf_file, " failed");
  }
  graph_def_ = *(reinterpret_cast<const xdl::proto::GraphDef*>(config.config()));

  // Step 1. mapping convert
  XdlSparse2Blaze();

  // Step 2: merge dense graph
  MergeDenseGraph();

  // Step 3. optimization pass
  SparseOptimizationPass();

  // Step 4: deactivate indepency node.
  Graph graph(net_def_);
  graph.DeactivateIndependency(net_def_.external_output(0).name());
  net_def_ = graph.GetNetDef();
}

void XdlImporter::XdlSparse2Blaze(){
  // The input name of XDL Raw Graph is not the real input name, should use
  // "tag" for name transformation.
  InitSparseInputAliasMap();

  // The input name of BackendOp in XDL Raw Graph is not the real input name,
  // should use "var_name_str" for name transformation.
  InitDenseInputAliasMap();

  // Start create op node one by one.
  CreateOpNode();
}

void XdlImporter::SparseOptimizationPass() {
  net_def_ = Optimizer::Get()->RunPass(net_def_, "XdlSparseFusionPass");
}

void XdlImporter::InitSparseInputAliasMap() {
  for (const auto& input : graph_def_.tag().input()) {
    sparse_input_alias_map_[input.op_name()] = input.input_name();
  }
}

void XdlImporter::InitDenseInputAliasMap() {
  for (const auto& node : graph_def_.node()) {
    if (node.op() == "MxnetBackendOp") {
      const auto& iter = node.attr().find("var_name_str");
      CHECK_TRUE(iter != node.attr().end(), "var_name_str is not found");

      const std::string& var_name_str= iter->second.s();
      std::vector<std::string> dense_input_names = Split(var_name_str, kVarNameDelim);
      // no need to convert last constant input
      BLAZE_CONDITION_THROW(dense_input_names.size() == node.input_size() - 1,
                            " dense input names size should equal node input size - 1!",
                            " dense_input_names.size()=", dense_input_names.size(),
                            " node.input_size()=", node.input_size());
      for (auto i = 0; i < dense_input_names.size(); ++i) {
        dense_input_alias_map_[node.input(i)] = dense_input_names[i];
        LOG_DEBUG("%s  %s", dense_input_names[i].c_str(), node.input(i).c_str());
      }
    } // TODO: Add TensorFlowBackendOp
  }
}

void XdlImporter::CreateOpNode() {
  for (const auto& node : graph_def_.node()) {
    const auto& iter = op_process_func_map_.find(node.op());
    if (iter == op_process_func_map_.end()) {
      LOG_ERROR("layer: %s is not registered", node.op().c_str());
      continue;
    }
    iter->second(node);
  
    // MxnetBackendOp is the last op of inference graph
    if (node.op() == "MxnetBackendOp") break;
    // TODO: Add TensorFlowBackendOp
  }

  // Set external output value info.
  auto output = net_def_.add_external_output();
  output->set_name(graph_def_.tag().output().op_name());
  output->set_dtype(data_type());
}

std::string XdlImporter::GetSparseInputAliasName(const std::string& name) const {
  const auto& iter = sparse_input_alias_map_.find(name);
  if (iter == sparse_input_alias_map_.end()) {
    return name;
  } else {
    return iter->second;
  }
}

std::string XdlImporter::GetDenseInputAliasName(const std::string& name) const {
  const auto& iter = dense_input_alias_map_.find(name);
  if (iter == dense_input_alias_map_.end()) {
    return name;
  } else {
    return iter->second;
  }
}

void XdlImporter::GenOpInputOutputInfo(const xdl::proto::NodeDef &node,
                                       size_t output_size,
                                       OperatorDef *op) {
  for (auto k = 0; k < node.input_size(); ++k) {
    op->add_input(GetSparseInputAliasName(node.input(k)));
  }
  for (auto k = 0; k < output_size; ++k) {
    const std::string output_name = node.name() + ":" + std::to_string(k);
    op->add_output(GetDenseInputAliasName(output_name));
  }
}

blaze::DataType XdlImporter::DeduceInputDataType(const std::string& input_name) {
  auto found = input_name.rfind(kDelimiter);
  if (found != std::string::npos) {
    std::string suffix = input_name.substr(found + 1);
    std::string prefix = input_name.substr(0, found);
    if (suffix == "values") {
      return data_type();
    } else if (suffix == "segments") {
      return blaze::kInt32;
    } else if (suffix == "indices") {
      return blaze::kInt32;
    } else if (suffix == "ids") {
      return blaze::kInt64;
    } else if (prefix == "indicator") {
      return blaze::kInt32;
    }
  }
  BLAZE_THROW("invalid input name: %s", input_name);
}

void XdlImporter::ProcessGetBatchOp(const xdl::proto::NodeDef &node) {
  // set external inputs
  std::set<std::string> external_inputs;
  for (const auto& input : graph_def_.tag().input()) {
    if (input.input_name() != "skbuf"
        && input.input_name() != "sklen"
        && input.input_name() != "label"
        && input.input_name().find(".indices") == std::string::npos) {
    } else { continue; }

    ValueInfo* value_info = net_def_.add_external_input();
    value_info->set_name(input.input_name());
    value_info->set_feature_name(input.input_name());
    
    // deduce datatype
    value_info->set_dtype(DeduceInputDataType(input.input_name()));
    value_info->set_level(input.table());

    if (input.input_name().find(kIndicatorPrefix) != std::string::npos) {
      // is indicator
      auto splits = Split(input.input_name(), kSparseFeatureSep);
      int level = std::stoi(splits[1].c_str());
      value_info->set_level(level);
      value_info->set_input_type(kInputIndicator);
      value_info->set_feature_name(GetSparseFeatureName(input.input_name()));
    } else {
      switch (value_info->input_type()) {
        case xdl::proto::kSparse:
          // is sparse feature, may be ids/values/segments.
          value_info->set_input_type(GetSparseInputType(input.input_name()));
          value_info->set_feature_name(GetSparseFeatureName(input.input_name()));
          break;
        case xdl::proto::kDense:
          value_info->set_input_type(kInputDense);
          break;
      }
    }
  }
}

void XdlImporter::ProcessPsIsInitializedOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsIsInitializedOp");
}

void XdlImporter::ProcessPsNormalInitializerOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("ProcessPsNormalInitializerOp");
}

void XdlImporter::ProcessPsRegisterVariableOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsRegisterVariableOp");

  const auto& iter = node.attr().find("var_name");
  if (iter != node.attr().end()) {
    // parse var name, which is the ps table name
    const std::string& var_name = iter->second.s();
    // parse dim size
    const auto& shape_iter = node.attr().find("shape");
    if (shape_iter != node.attr().end()) {
      const auto& shape = shape_iter->second.shape();
      if (shape.dim_size() > 1) {
        var_dim_map_[var_name] = shape.dim(1);
      }
    }
  }
}

void XdlImporter::ProcessPsConstantInitializerOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsConstantInitializerOp");
}

void XdlImporter::ProcessPsTruncatedNormalInitializerOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsTruncatedNormalInitializerOp");
}

void XdlImporter::ProcessPsIdentityInitializerOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsIdentityInitializerOp");
}

void XdlImporter::ProcessUniqueOp(const xdl::proto::NodeDef &node) {
  CHECK_EQ(node.input_size(), 2, "node.input_size()=", node.input_size());

  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("Unique");
  GenOpInputOutputInfo(node, kUniqueOpOutputSize, op);
}

void XdlImporter::ProcessTakeOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("Gather");
  GenOpInputOutputInfo(node, kTakeOpOutputSize, op);
}

void XdlImporter::ProcessPsSparsePullOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsSparsePullOp");
  GenOpInputOutputInfo(node, kPsSparsePullOpOutputSize, op);

  const auto& iter = node.attr().find("var_name");
  if (iter != node.attr().end()) {
    // set var name
    const std::string &var_name = iter->second.s();
    blaze::Argument* var_name_arg = op->add_arg();
    var_name_arg->set_name("var_name");
    var_name_arg->set_s(var_name);

    // set dim
    const auto& dim_iter = var_dim_map_.find(var_name);
    CHECK_NE(dim_iter, var_dim_map_.end(), "var_name:", var_name, " sparse dim not configed!");

    blaze::Argument* dim_arg = op->add_arg();
    dim_arg->set_name("dim");
    dim_arg->set_i(dim_iter->second);
  }
}

void XdlImporter::ProcessPsPullOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("PsPullOp");
  GenOpInputOutputInfo(node, kPsPullOpOutputSize, op);
}

void XdlImporter::ProcessConstantOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("_Constant");
  GenOpInputOutputInfo(node, kConstantOutputSize, op);
}

void XdlImporter::ProcessKSumOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("KSum");
  bool average = false;
  const auto& iter = node.attr().find("average");
  if (iter != node.attr().end()) {
    average = iter->second.b();
  }
  if (average) {
    auto average_arg = op->add_arg();
    average_arg->set_name("average");
    average_arg->set_i(static_cast<int64_t>(average));
  }
  GenOpInputOutputInfo(node, kKSumOutputSize, op);
}

void XdlImporter::ProcessTileOp(const xdl::proto::NodeDef &node) {
  auto op = net_def_.add_op();
  op->set_name(node.name());
  op->set_type("Tile");

  // get trunc direction and length
  bool reverse = false;
  const auto& reverse_iter = node.attr().find("reverse");
  if (reverse_iter != node.attr().end()) {
    reverse = reverse_iter->second.b();
  }
  int length = 0;
  const auto& len_iter = node.attr().find("length");
  if (len_iter != node.attr().end()) {
    length = reverse_iter->second.i();
  }

  // write operator def info
  auto reverse_arg = op->add_arg();
  reverse_arg->set_name("reverse");
  reverse_arg->set_i(static_cast<int64_t>(reverse));
  auto len_arg = op->add_arg();
  len_arg->set_name("length");
  len_arg->set_i(static_cast<int64_t>(length));

  GenOpInputOutputInfo(node, kTileOutputSize, op);
}

void XdlImporter::ProcessMxnetBackendOp(const xdl::proto::NodeDef &node) {
  const auto& iter = node.attr().find("graph_def");
  CHECK_NE(iter, node.attr().end(), "graph_def is not defined in MxnetBackendOp");
  backend_model_conf_ = iter->second.s();
  backend_type_ = "mxnet";
}

void XdlImporter::ProcessTFBackendOp(const xdl::proto::NodeDef& node) {
  const auto& iter = node.attr().find("graph_def");
  CHECK_NE(iter, node.attr().end(), "graph_def is not defined in TFBackendOp");
  backend_model_conf_ = iter->second.s();
  backend_type_ = "tensorflow";
}

void XdlImporter::MergeDenseGraph() {
  CHECK_TRUE(backend_model_conf_.size() > 0, "backend model conf is empty");
  NetDef dense_net_def;

  if (backend_type_ == "mxnet") {
    std::istringstream is(backend_model_conf_);
    mxnet_importer_.LoadModel(is, data_file_);
    LOG_DEBUG("mxnet_importer=%s", mxnet_importer_.net_def().DebugString().c_str());
    dense_net_def = mxnet_importer_.net_def();
  } else {
    tensorflow_importer_.LoadModelFromString(backend_model_conf_, data_file_);
    LOG_DEBUG("tensorflow_importer=%s", tensorflow_importer_.net_def().DebugString().c_str());
    dense_net_def = tensorflow_importer_.net_def();
  }
  for (const auto& op : dense_net_def.op()) {
    auto new_op = net_def_.add_op();
    *new_op = op;
  }
}

void XdlImporter::SetProcessNodeFunction(const std::string &name, ProcessOpNodeFunction function) {
  op_process_func_map_[name] = function;
}

}  // namespace blaze
