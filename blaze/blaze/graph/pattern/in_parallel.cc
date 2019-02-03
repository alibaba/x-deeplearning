/*
 * \file in_parallel.h 
 * \brief The in parallel pattern Match.
 */
#include "blaze/graph/fusion_pattern.h"

namespace blaze {

static bool SearchGraph(Node& node, void* arg) {
  std::pair<FusionPattern*, Graph*>* pair =
      reinterpret_cast<std::pair<FusionPattern*, Graph*>*>(arg);
  FusionPattern* pattern = pair->first;
  Graph* graph = pair->second;

  const std::vector<int>& pattern_output = pattern->output();
  const std::vector<int>& pattern_input = pattern->input();
  const auto& pattern_def = pattern->pattern_def();

  const std::string& pattern_output_type = pattern_def.node(pattern_output[0]).type();
  if (node.op.type() != pattern_output_type) return false;

  std::vector<ArgumentHelper*> args;
  std::vector<Node*> nodes;
  if (pattern_def.node_size() > 1) {
    // compare parents.
    const std::string& pattern_input_type = pattern_def.node(pattern_input[0]).type();
    bool hit_parent = false;
    for (const auto& parent_iter : node.parents) {
      int parent_idx = parent_iter.first;
      Node& parent_node = graph->node(parent_idx);
      if (parent_node.op.type() == pattern_input_type) {
        hit_parent = true;
        args.push_back(new ArgumentHelper(parent_node.op));
        nodes.push_back(&parent_node);
        break;
      }
    }
    if (!hit_parent) return false;
  }
  args.push_back(new ArgumentHelper(node.op));
  nodes.push_back(&node);

  bool success = true;
  if (pattern->fusion_pattern_impl() != nullptr) {
    pattern->fusion_pattern_impl()->Init();
    if (!pattern->fusion_pattern_impl()->Match(args, nodes, graph)) {
      success = false;
    } else {
      pattern->fusion_pattern_impl()->GraphRewrite(args, nodes, graph);
    }
  }
  for (auto& argument_helper : args) {
    delete argument_helper;
  }
  return success;
}

static bool InParallelMatch(FusionPattern* pattern, Graph* graph) {
  std::pair<FusionPattern*, Graph*> pair(pattern, graph);
  return graph->BFS(SearchGraph, &pair);
}

REGISTER_FMATCH(kInParallel, InParallelMatch);

}  // namespace blaze

