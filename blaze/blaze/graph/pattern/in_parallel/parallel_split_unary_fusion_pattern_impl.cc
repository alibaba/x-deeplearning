/*
 * \file parallel_split_unary_fusion_pattern_impl.cc 
 * \brief The parallel split unary fusion pattern.
 * Such as: Split->Softmax/Softmax/Softmax/... -> Softmax/Split
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_unary_fusion_pattern_impl.h"

#include <sstream>

namespace blaze {

void ParallelSplitUnaryFusionPatternImpl::Init() {
  candidate_idx_.clear();
}

bool ParallelSplitUnaryFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                                const std::vector<Node*>& nodes,
                                                Graph* graph) {
  if (args[0]->GetSingleArgument<int>("axis", 1) != 0) return false;

  Node* split_node = nodes[0];
  Node* unary_node = nodes[1];
  // check the unary op can be fused?
  if (!UnaryNodeMatch(args[1])) return false;
  // The children size equals output size
  if (split_node->children.size() != split_node->op.output_size()) return false;

  candidate_idx_.resize(split_node->children.size());
  for (const auto& child : split_node->children) {
    Node& child_node = graph->node(child.first);
    if (child_node.op.type() != unary_node->op.type()) return false;
    ArgumentHelper argument_helper(child_node.op);
    // check two unary op can be fused?
    if (!UnaryNodeMatch(args[1], &argument_helper)) return false;
    int output_idx = split_node->GetOutputIdx(child_node.op.input(0));
    candidate_idx_[output_idx] = child.first;
  }
  split_node_idx_ = split_node->idx;
  if (candidate_idx_.size() > 1) return true;
  return false;
}
  
void ParallelSplitUnaryFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                                       std::vector<Node*>& nodes,
                                                       Graph* graph) {
  OperatorDef unary_op, split_op;

  // Deactivate candidate Subgraph
  graph->DeactivateSubgraph(candidate_idx_);
  // Deactivate split Subgraph
  std::vector<int> subgraph;
  subgraph.push_back(split_node_idx_);
  graph->DeactivateSubgraph(subgraph);

  Node& n = graph->node(candidate_idx_[0]);
  // set the name of unary_op
  std::stringstream name;
  name << n.op.name() << "_parallel";
  unary_op.set_name(name.str());
  // set type of unary_op
  if (this->pattern->fusion_op_name().empty()) {
    unary_op.set_type(nodes[1]->op.type());
  } else {
    unary_op.set_type(this->pattern->fusion_op_name());
  }
  // set argument of unary op
  UpdateFusedUnaryArg(args[1], &unary_op);
  // set input of unary_op
  unary_op.add_input(nodes[0]->op.input(0));
  // set output of unary_op
  unary_op.add_output(name.str());

  // set the name of split_op
  split_op.set_name(name.str());
  // set type of split_op
  split_op.set_type(nodes[0]->op.type());
  // set input of split_op
  split_op.add_input(name.str());
  // set argument of split op
  Argument* arg = split_op.add_arg();
  arg->set_name("axis");
  arg->set_i(0);
  // set output of split op
  for (const auto& idx : candidate_idx_) {
    Node& nn = graph->node(idx);
    for (const auto& oname : nn.op.output()) {
      split_op.add_output(oname);
    }
  }

  // insert fused node
  graph->InsertNode(unary_op);
  graph->InsertNode(split_op);
}

}  // namespace blaze

