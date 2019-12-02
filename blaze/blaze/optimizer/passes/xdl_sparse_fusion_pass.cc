/*!
 * \file xdl_sparse_fusion_pass.cc
 * \brief The fusion pass for xdl sparse model.
 */
#include "blaze/optimizer/passes/xdl_sparse_fusion_pass.h"
#include "blaze/operator/sparse_op/embedding/embedding_op.h"

namespace blaze {

XdlSparseFusionPass& XdlSparseFusionPass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

XdlSparseFusionPass& XdlSparseFusionPass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef XdlSparseFusionPass::RunPass(const NetDef& net_def) {
  bool conti = true;
  NetDef ret = net_def;
  while (conti) {
    Graph graph(ret, false);
    Graph* graph_ptr = &graph;
    conti = graph.BFS([this, graph_ptr](Node& node, void* arg) {
      return this->SparseNodeFusionPass(graph_ptr, node, arg);
    }, nullptr);
    ret = graph.GetNetDef();
  }
  return ret;
}

bool XdlSparseFusionPass::SparseNodeFusionPass(Graph* graph, Node& node, void* arg) {
  if (node.op.type() == "Unique") {
    return UniqueNodeEliminatePass(graph, node, arg);
  } else if (node.op.type() == "PsPullOp") {
    return PsPullNodeEliminatePass(graph, node, arg);
  } else if (node.op.type() == "PsSparsePullOp") {
    return PsSparsePullNodeReplacePass(graph, node, arg);
  }
  return false;
}

bool XdlSparseFusionPass::UniqueNodeEliminatePass(Graph* graph, Node& node, void* arg) {
  BLAZE_CONDITION_THROW(node.op.input_size() == 2, "node.op.input_size()=", node.op.input_size());
  BLAZE_CONDITION_THROW(node.op.output_size() == 4, "node.op.output_size()=", node.op.output_size());
  const auto& input_name = node.op.input(0);
  const auto& first_output = node.op.output(0);
  const auto& second_output = node.op.output(1);
  // change children input.
  // record input node idx and process later because
  // rename & remove input will reconstruct children list.
  const auto& children = node.children;
  std::set<int> to_rename_children;
  std::set<int> to_remove_input_children;
  for (const auto& item : children) {
    for (const auto& name : item.second) {
      if (name == first_output) {
        to_rename_children.insert(item.first);
      } else if (name == second_output) {
        to_remove_input_children.insert(item.first);
      }
    }
  }
  auto node_idx = node.idx;
  for (auto child_idx : to_rename_children) {
    graph->RenameInput(child_idx, first_output, input_name);
  }
  for (auto child_idx : to_remove_input_children) {
    graph->RemoveInput(child_idx, second_output);
  }
  // deactive unique node, can not refer node.
  graph->DeactivateSubgraph({ node_idx });
  return true;
}

bool XdlSparseFusionPass::PsPullNodeEliminatePass(Graph *graph, Node &node, void *arg) {
  BLAZE_CONDITION_THROW(node.op.output_size() == 1, "node.op.output_size()=", node.op.output_size());

  // if PsPullOp node has no child, no need to eliminate in this pass
  const auto& children = node.children;
  if (children.size() == 0) return false;

  // remove children input
  const auto& output_name = node.op.output(0);
  std::set<int> to_remove_input_children;
  for (const auto& item : children) {
    for (const auto &name : item.second) {
      if (name == output_name) {
        to_remove_input_children.insert(item.first);
      }
    }
  }
  for (auto child_idx : to_remove_input_children) {
    graph->RemoveInput(child_idx, output_name);
  }

  // deactive pspull node
  graph->DeactivateSubgraph({node.idx});
  return true;
}

static bool IsEmbeddingOp(const std::string& op_type) {
  static std::set<std::string> embed_op_sets = {
    "KSum",
    "Tile"
  };
  return embed_op_sets.count(op_type);
}

static void ResetInputOutput(const Node& child_node, OperatorDef* new_op) {
  static std::map<std::string, std::vector<int>> in_direction = {
    { "KSum" , { 1, 2 } },  // 0: uniq_id 1: value 2: segments
    { "Tile" , { 1, 2 } },  // 0: uniq_id 1: value 2: segments
  };
  static std::map<std::string, std::vector<int>> out_direction = {
    { "KSum" , { 0 } },
    { "Tile" , { 0 } }
  };
  for (auto index : in_direction[child_node.op.type()]) {
    new_op->add_input(child_node.op.input(index));
  }
  for (auto index : out_direction[child_node.op.type()]) {
    new_op->add_output(child_node.op.output(index));
  }
}

static UDFType GetUDFType(const Node& node) {
  if (node.op.type() == "Tile") {
    return UDFType::kAssign;
  } else if (node.op.type() == "KSum") {
    ArgumentHelper ksum_helper(node.op);
    auto average = ksum_helper.GetSingleArgument<bool>("average", false);
    if (average)
      return UDFType::kAvg;
    else
      return UDFType::kSum;
  } else {
    BLAZE_THROW("Unexpected op type: ", node.op.type());
  }
}

bool XdlSparseFusionPass::PsSparsePullNodeReplacePass(Graph* graph, Node& node, void* arg) {
  BLAZE_CONDITION_THROW(node.op.input_size() == 2, "node.op.input_size()=", node.op.input_size());
  std::vector<int> todel_subgraph;
  std::vector<OperatorDef> toadd_ops;
  todel_subgraph.push_back(node.idx);
  const auto& children = node.children;
  
  for (const auto& item : children) {
    auto child_idx = item.first;
    Node& child_node = graph->node(child_idx);
    if (!IsEmbeddingOp(child_node.op.type())) continue;
    
    BLAZE_CONDITION_THROW(child_node.op.input_size() >= 3, "child_node.op.input_size()=", child_node.op.input_size());
    BLAZE_CONDITION_THROW(child_node.op.output_size() == 1, "child_node.op.output_size()=", child_node.op.output_size());
    todel_subgraph.push_back(child_idx);
    
    // create embedding node
    OperatorDef new_op(node.op);
    new_op.clear_output();
    new_op.clear_arg();
    new_op.set_type("Embedding");
    new_op.set_name(child_node.op.name());

    //remove the 2nd input "save-ratio"
    const auto& const_name = new_op.input(1);
    int const_idx = node.GetParentIdx(const_name);
    todel_subgraph.push_back(const_idx);
    new_op.mutable_input()->erase(new_op.input().begin()+1);

    // set device_option
    new_op.mutable_device_option()->set_device_type(kCPU);
    new_op.mutable_device_option()->set_device_id(0);

    // reset embedding input and output
    ResetInputOutput(child_node, &new_op);
    // generate embedding args
    GenEmbeddingArgs(node, child_node, &new_op);

    // add embedding op
    toadd_ops.push_back(new_op);
  }
  // deactive subgraph
  graph->DeactivateSubgraph(todel_subgraph);
  // insert embedding node
  for (const auto& op : toadd_ops) {
    graph->InsertNode(op);
  }
  return true;
}

void XdlSparseFusionPass::GenEmbeddingArgs(const Node& node, const Node& child_node, OperatorDef* op_def) {
  // generate embedding config pb
  blaze::EmbeddingConfig embedding_config;
  auto fg_config = embedding_config.add_feature_group_config();
  
  const auto& ids_input_name = node.op.input(0);
  std::string fg_name = ids_input_name.substr(0, ids_input_name.rfind(kIdSuffix));
  fg_config->set_feature_group(fg_name);
  
  std::string var_name = ArgumentHelper::GetSingleArgument<OperatorDef, std::string>(node.op, "var_name", "");
  fg_config->set_table_name(var_name);
  
  int dim = ArgumentHelper::GetSingleArgument<OperatorDef, int>(node.op, "dim", 0);
  fg_config->set_dim(dim);
  
  auto block_config = embedding_config.add_block_config();
  auto block_config_item = block_config->add_embedding_block_config_item();
  block_config_item->set_feature_group(fg_name);
  block_config_item->set_udf_type(GetUDFType(child_node));
  if (block_config_item->udf_type() == UDFType::kAssign) {
    bool reverse = ArgumentHelper::GetSingleArgument<OperatorDef, bool>(child_node.op, "reverse", false);
    int length = ArgumentHelper::GetSingleArgument<OperatorDef, int>(child_node.op, "length", 0);
    block_config_item->set_trunc_direction(reverse ? TruncDirection::kReverse : TruncDirection::kInorder);
    block_config_item->set_trunc_num(length);
  }

  auto embedding_config_arg = op_def->add_arg();
  embedding_config_arg->set_name("embedding_config");
  embedding_config_arg->set_s(embedding_config.DebugString());
}

}  // namespace blaze


