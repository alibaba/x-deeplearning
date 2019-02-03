/*
 * \file in_order.h 
 * \brief The in order pattern Match.
 */
#include "blaze/graph/fusion_pattern.h"

#include "blaze/common/common_defines.h"
#include "blaze/common/proto_helper.h"

namespace blaze {

// Inorder match is a sequence Operator matching, each operator has one output.
// And now we only support length-2 sequence operators pattern.
// in-order pattern has only one successor.
static bool SearchGraph(Node& node, void* arg) {
  std::pair<FusionPattern*, Graph*>* pair =
      reinterpret_cast<std::pair<FusionPattern*, Graph*>*>(arg);
  FusionPattern* pattern = pair->first;
  Graph* graph = pair->second;

  const std::vector<int>& pattern_output = pattern->output();
  const std::vector<int>& pattern_input = pattern->input();
  
  const std::string& pattern_output_type =
      pattern->pattern_def().node(pattern_output[0]).type();
  if (node.op.type() != pattern_output_type) return false;
  //if (node.parents.size() != node.op.input_size()) return false;

  for (const auto& parent_iter : node.parents) {
    int parent_idx = parent_iter.first;
    if (parent_iter.second.size() != 1) continue;

    Node& parent_node = graph->node(parent_idx);
    // in-order pattern has only one successor.
    if (parent_node.op.output_size() != 1) continue;

    const std::string& pattern_input_type =
        pattern->pattern_def().node(pattern_input[0]).type();
    if (parent_node.op.type() != pattern_input_type) continue;
      
    // check the argument is satisfied by the fusion pattern.
    ArgumentHelper argument_helper(node.op);
    ArgumentHelper parent_argument_helper(parent_node.op);
    std::vector<ArgumentHelper*> args;
    args.push_back(&parent_argument_helper);
    args.push_back(&argument_helper);

    std::vector<Node*> nodes;
    nodes.push_back(&parent_node);
    nodes.push_back(&node);
      
    if (pattern->fusion_pattern_impl() != nullptr) {
      pattern->fusion_pattern_impl()->Init();
      if (!pattern->fusion_pattern_impl()->Match(args, nodes, graph)) continue;
    }

    // gen the fused op arguments, and remove useless dependencies.
    node.op.set_type(pattern->fusion_op_name());

    if (pattern->fusion_pattern_impl() != nullptr) {
      pattern->fusion_pattern_impl()->GraphRewrite(args, nodes, graph);
    }
    // reset input for fused op.
    bool disable_reset_input = pattern->option("disable_reset_input");
    if (!disable_reset_input) {  // if enable reset input automatically
      std::vector<std::string> node_op_input;
      for (size_t k = 0; k < node.op.input_size(); ++k) {
        node_op_input.push_back(node.op.input(k));
      }
      const std::string& parent_node_op_output = parent_node.op.output(0);
      node.op.clear_input();
      for (const auto& name : node_op_input) {
        if (name != parent_node_op_output) {
          node.op.add_input(name);
        } else {
          for (const auto& item : parent_node.op.input()) {
            node.op.add_input(item);
          }
        }
      }
    }
     
    // parent node remove
    std::vector<int> subgraph;
    subgraph.push_back(parent_node.idx);
    auto subgraph_output = graph->GetSubgraphOutput(subgraph);
    if (subgraph_output.size() == 1) {
      graph->DeactivateSubgraph(subgraph);
    }
    return true;
  }
  return false;
}

static bool InorderMatch(FusionPattern* pattern, Graph* graph) {
  std::pair<FusionPattern*, Graph*> pair(pattern, graph);
  return graph->BFS(SearchGraph, &pair);
}

REGISTER_FMATCH(kInOrder, InorderMatch);

}  // namespace blaze

