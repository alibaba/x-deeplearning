/*
 * \file parallel_split_binary_fusion_pattern_impl.cc 
 * \brief The parallel split binary fusion pattern.
 * Such as: split->Binary/Binary/Binary/...
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_binary_fusion_pattern_impl.h"

#include <sstream>

#include "blaze/operator/common_helper.h"

namespace blaze {

void ParallelSplitBinaryFusionPatternImpl::Init() {
  binary_idx_.clear();
}

bool ParallelSplitBinaryFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                                 const std::vector<Node*>& nodes,
                                                 Graph* graph) {
  Node* node = nodes[1];
  int a_idx = node->GetParentIdx(node->op.input(0));
  int b_idx = node->GetParentIdx(node->op.input(1));
  if (a_idx < 0 || b_idx < 0) return false;

  Node& a_node = graph->node(a_idx);
  Node& b_node = graph->node(b_idx);
  if (a_node.op.type() != "Split" || b_node.op.type() != "Split") return false;

  ArgumentHelper a_node_helper(a_node.op);
  ArgumentHelper b_node_helper(b_node.op);
  if (a_node_helper.GetSingleArgument<int>("axis", 1) != 0) return false;
  if (a_node_helper.GetSingleArgument<int>("axis", 1) != 0) return false;

  // calculate all the possible binary ids for fusion.
  std::set<int> a_ids_set, and_ids;
  for (const auto& item : a_node.children) {
    Node& n = graph->node(item.first);
    if (n.op.type() == node->op.type()) {
      ArgumentHelper temp(n.op);
      if (BinaryNodeMatch(args[1], &temp)) {
        a_ids_set.insert(item.first);
      }
    }
  }
  a_ids_set.erase(node->idx);

  for (const auto& item : b_node.children) {
    Node& n = graph->node(item.first);
    if (n.op.type() == node->op.type() && a_ids_set.count(n.idx)) {
      and_ids.insert(item.first);
    }
  }
  binary_idx_.push_back(node->idx);
  // scan all the split node.
  int a_output_idx = a_node.GetOutputIdx(node->op.input(0));
  int b_output_idx = b_node.GetOutputIdx(node->op.input(1));

  int a_output_idx_end = a_output_idx + 1, b_output_idx_end = b_output_idx + 1;
  for (; a_output_idx_end < a_node.op.output_size() &&
       b_output_idx_end < b_node.op.output_size();
       a_output_idx_end++, b_output_idx_end++) {
    // find the matched node, whose parents are a/b and input locations are
    // sequential.
    int matched_idx = FindMatchedNode(graph, and_ids, a_node.idx, a_output_idx_end,
                                      b_node.idx, b_output_idx_end);
    if (matched_idx < 0) break;
    binary_idx_.push_back(matched_idx);
    and_ids.erase(matched_idx);
  }

  int a_output_idx_begin = a_output_idx - 1, b_output_idx_begin = b_output_idx - 1;
  for (; a_output_idx_begin >=0 && b_output_idx_begin >= 0;
       a_output_idx_begin--, b_output_idx_begin--) {
    // find the matched node, whose parents are a/b and input locations are
    // sequential.
    int matched_idx = FindMatchedNode(graph, and_ids, a_node.idx, a_output_idx_begin,
                                      b_node.idx, b_output_idx_begin);
    if (matched_idx < 0) break;
    binary_idx_.insert(binary_idx_.begin(), matched_idx);
    and_ids.erase(matched_idx);
  }
  if (a_output_idx_end - a_output_idx_begin <= 2) return false;

  a_node_idx_ = a_node.idx;
  b_node_idx_ = b_node.idx;
  a_node_output_begin_ = a_output_idx_begin + 1;
  a_node_output_end_ = a_output_idx_end;
  b_node_output_begin_ = b_output_idx_begin + 1;
  b_node_output_end_ = b_output_idx_end;
  //transA_ = transA;
  //transB_ = transB;

  // we don't consider recursively adjust the layout
  if (a_node.op.output_size() % (a_node_output_end_ - a_node_output_begin_) != 0) return false;
  if (a_node_output_begin_ % (a_node_output_end_ - a_node_output_begin_) != 0) return false;
  if (b_node.op.output_size() % (b_node_output_end_ - b_node_output_begin_) != 0) return false;
  if (b_node_output_begin_ % (b_node_output_end_ - b_node_output_begin_) != 0) return false;

  return true;
}
  
void ParallelSplitBinaryFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                                        std::vector<Node*>& nodes,
                                                        Graph* graph) {
  // add new fusion node/splt node. and remove useless node
  OperatorDef fmt, fsa, fsb, split_node;

  bool create_a_flag = InitFusedSplitOp(fsa, a_node_idx_, a_node_output_begin_, a_node_output_end_, graph);
  bool create_b_flag = InitFusedSplitOp(fsb, b_node_idx_, b_node_output_begin_, b_node_output_end_, graph);
  // Deactivate BinaryIdx Subgraph
  graph->DeactivateSubgraph(binary_idx_);

  // set input of fmt.
  int a_output_idx = a_node_output_begin_ / (a_node_output_end_ - a_node_output_begin_);
  int b_output_idx = b_node_output_begin_ / (b_node_output_end_ - b_node_output_begin_);
  fmt.add_input(fsa.output(a_output_idx));
  fmt.add_input(fsb.output(b_output_idx));
  // init parallel binary node.
  InitFusedBinaryOp(fmt, split_node, binary_idx_, args[1], graph);

  if (create_a_flag) graph->InsertNode(fsa);
  if (create_b_flag) graph->InsertNode(fsb);
  graph->InsertNode(fmt);
  graph->InsertNode(split_node);

  // check and deactivate split node.
  graph->CheckAndDeactivateNode(a_node_idx_);
  graph->CheckAndDeactivateNode(b_node_idx_);
}

int ParallelSplitBinaryFusionPatternImpl::FindMatchedNode(Graph* graph,
                                                          const std::set<int>& ids,
                                                          int a_node_idx,
                                                          int a_output_idx,
                                                          int b_node_idx,
                                                          int b_output_idx) {
  for (const auto& id : ids) {
    Node& node = graph->node(id);
    int cur_a_idx = node.GetParentIdx(node.op.input(0));
    int cur_b_idx = node.GetParentIdx(node.op.input(1));
    Node& a_node = graph->node(cur_a_idx);
    Node& b_node = graph->node(cur_b_idx);
    if (a_node.idx != a_node_idx || b_node.idx != b_node_idx) continue;
    int cur_a_output_idx = a_node.GetOutputIdx(node.op.input(0));
    int cur_b_output_idx = b_node.GetOutputIdx(node.op.input(1));
    if (a_output_idx != cur_a_output_idx || b_output_idx != cur_b_output_idx) continue;
    return id;
  }
  return -1;
}

bool ParallelSplitBinaryFusionPatternImpl::InitFusedSplitOp(OperatorDef& op,
                                                            int node_idx,
                                                            int begin,
                                                            int end,
                                                            Graph* graph) {
  Node& n = graph->node(node_idx);

  // Find the siblings for the available Split node, if exist, Return
  // flase and copy the sibling's output names.
  int match_output_size = n.op.output_size() / (end - begin);
  int match_output_idx = begin / (end - begin); 
  
  int sibling_node_idx = graph->FindSibling(node_idx, n.op.input(0), "Split", match_output_size);
  if (sibling_node_idx >= 0) {
    Node& sibling_node = graph->node(sibling_node_idx);
    for (const auto& oname : sibling_node.op.output()) {
      op.add_output(oname);
    }
    return false;
  }

  // set name of op
  static int transform_id = 1;
  std::stringstream name;
  name << n.op.name() << "_" << transform_id++;
  op.set_name(name.str());
  // set type of op
  op.set_type("Split");
  // set argument of op
  Argument* arg = op.add_arg();
  arg->set_name("axis");
  arg->set_i(0);
  // set input of op
  op.add_input(n.op.input(0));
  // set output of op
  name << "_output";
  size_t split_num = n.op.output_size() / (end - begin);
  for (size_t k = 0; k < split_num; ++k) {
    std::stringstream output_name;
    output_name << name.str() << "_" << k;
    op.add_output(output_name.str());
  }
  return true;
}

void ParallelSplitBinaryFusionPatternImpl::InitFusedBinaryOp(OperatorDef& op,
                                                             OperatorDef& split_node,
                                                             const std::vector<int>& binary_idx,
                                                             const ArgumentHelper* arg_node1,
                                                             Graph* graph) {
  Node& n = graph->node(binary_idx[0]);
  // set name of op
  static int transform_id = 1;
  std::stringstream name;
  name << n.op.name() << "_" << transform_id;
  op.set_name(name.str());
  // set type of fusion op
  op.set_type(this->pattern->fusion_op_name());
  // set argument of op
  UpdateFusedBinaryArg(arg_node1, &op);
  auto arg = op.add_arg();
  arg->set_name(kOpArgNameParallelNum);
  arg->set_i(binary_idx.size());
  // set output of op
  op.add_output(name.str());

  // set name of split_node.
  std::stringstream split_name;
  split_name << n.op.name() << "_split_" <<  transform_id++;
  split_node.set_name(split_name.str());
  // set type of split_node
  split_node.set_type("Split");
  // set argument of split node
  arg = split_node.add_arg();
  arg->set_name("axis");
  arg->set_i(0);
  // set input of split_node.
  split_node.add_input(name.str());
  // set output of split_node
  for (const auto& idx : binary_idx) {
    Node& nn = graph->node(idx);
    for (const auto& oname : nn.op.output()) {
      split_node.add_output(oname);
    }
  }
}

}  // namespace blaze
