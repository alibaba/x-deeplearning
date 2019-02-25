/*!
 * \file eliminate_pass.cc
 * \brief The eliminate pass for eliminating useless op.
 */
#include "blaze/optimizer/passes/eliminate_pass.h"

#include "blaze/common/proto_helper.h"

namespace {
const float kAddEliminateMin = -1e-6;
const float kAddEliminateMax = 1e-6;
} // namespace

namespace blaze {

EliminatePass& EliminatePass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

EliminatePass& EliminatePass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef EliminatePass::RunPass(const NetDef& net_def) {
  bool conti = true;
  NetDef ret = net_def;
  while (conti) {
    Graph graph(ret);
    Graph* graph_ptr = &graph;
    conti = graph.BFS([this, graph_ptr](Node& node, void* arg) {
                return this->NodeEliminatePass(graph_ptr, node, arg);
            }, nullptr);
    ret = graph.GetNetDef();
  }
  return ret;
}

bool EliminatePass::NodeEliminatePass(Graph* graph, Node& node, void* arg) {
  if (node.op.type() == "Add") {
    return AddNodeEliminatePass(graph, node, arg);
  }
  return false;
}

bool EliminatePass::AddNodeEliminatePass(Graph* graph, Node& node, void* arg) {
  BLAZE_CONDITION_THROW(node.op.input_size() == 2, "node.op.input_size()=", node.op.input_size());
  
  for (auto i = 0; i < node.op.input_size(); ++i) {
    int parent_id = node.GetParentIdx(node.op.input(i));
    Node& parent_node = graph->node(parent_id);
    if (parent_node.op.type() == "ConstantFill") {
      ArgumentHelper helper(parent_node.op);
      if (helper.ConstantValueInRange(kAddEliminateMin, kAddEliminateMax)) {
        std::vector<int> todel_subgraph;
        todel_subgraph.push_back(node.idx);
        std::vector<OperatorDef> toadd_ops;
        
        // TODO: Use graph's rename interface.
        const auto& children = node.children;
        for (const auto& item : children) {
          Node& used_node = graph->node(item.first);
          todel_subgraph.push_back(used_node.idx);
          const auto& name = item.second[0];
          std::vector<std::string> inputs;
          for (auto k = 0; k < used_node.op.input_size(); ++k) {
            if (used_node.op.input(k) == name) {
              inputs.push_back(node.op.input(1 - i));
            } else {
              inputs.push_back(used_node.op.input(k));
            }
          }
          OperatorDef new_op = used_node.op; 
          new_op.clear_input();
          for (const auto& item : inputs) new_op.add_input(item);
          toadd_ops.push_back(new_op);
        }
        graph->DeactivateSubgraph(todel_subgraph);
        for (const auto& op : toadd_ops) {
          graph->InsertNode(op);
        }
        return true;
      }
    }
  }
  return false;
}

}  // namespace blaze


