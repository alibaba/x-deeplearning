/*!
 * \file concat_reduce_swap_pass.cc
 * \brief The concat reduce swap pass
 */
#include "blaze/optimizer/passes/concat_reduce_swap_pass.h"

#include <sstream>

#include "blaze/common/proto_helper.h"
#include "blaze/operator/common_helper.h"

namespace blaze {

ConcatReduceSwapPass& ConcatReduceSwapPass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

ConcatReduceSwapPass& ConcatReduceSwapPass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef ConcatReduceSwapPass::RunPass(const NetDef& net_def) {
  // traverse the graph and swap Concat->ReduceSum ops. 
  bool conti = true;
  NetDef ret = net_def;
  while (conti) {
    Graph graph(ret);
    Graph* graph_dptr = &graph;
    conti = graph.BFS([this, graph_dptr](Node& node, void* arg) {
                return this->ConcatReduceNodeSwapPass(graph_dptr, node, arg);
            }, nullptr);
    ret = graph.GetNetDef();
  }
  return ret;
}

bool ConcatReduceSwapPass::ConcatReduceNodeSwapPass(Graph* graph, Node& node, void* arg) {
  if (node.op.type() != "Concat") return false;
  if (node.children.size() != 1) return false;  // must has one output
  int node_idx = node.idx;
  int child_idx = node.children.begin()->first;
  Node& child = graph->node(child_idx);
  if (!IsReduceOp(child)) return false;

  ArgumentHelper node_argument_helper(node.op);
  ArgumentHelper child_argument_helper(child.op);
  size_t concat_axis = node_argument_helper.GetSingleArgument<size_t>("axis", 1);
  size_t reduce_axis = CommonHelper::GetReduceAxis(&child_argument_helper);
  int keepdims = child_argument_helper.GetSingleArgument<int>("keepdims", 1);
  if (concat_axis <= reduce_axis) return false;

  // node is concat node, child is reduce node.
  int concat_input_size = node.op.input_size();
  std::vector<OperatorDef> reduce_ops;
  reduce_ops.resize(concat_input_size);
  OperatorDef concat_op = node.op;
  if (keepdims == 0) ArgumentHelper::SetSingleArgument<int>(concat_op, "axis", concat_axis - 1);
  // reset name of concat op
  std::stringstream concat_name;
  concat_name << concat_op.name() << "_0";
  concat_op.set_name(concat_name.str());
  // clear input of concat op
  concat_op.clear_input();
  // reset output of concat op
  concat_op.clear_output();
  concat_op.add_output(child.op.output(0));

  for (size_t k = 0; k < reduce_ops.size(); ++k) {
    reduce_ops[k] = child.op;  // is reduce op
    // reset name of reduce op
    std::stringstream name;
    name << reduce_ops[k].name() << "_" << k;
    reduce_ops[k].set_name(name.str());
    // reset input of reduce op
    reduce_ops[k].clear_input();
    reduce_ops[k].add_input(node.op.input(k));
    // reset output of reduce op
    reduce_ops[k].clear_output();
    reduce_ops[k].add_output(name.str());

    // add input for concat.
    concat_op.add_input(name.str());
  }
  // Deactivate Concat/Reduce node.
  std::vector<int> subgraph;
  subgraph.push_back(child_idx);
  subgraph.push_back(node_idx);
  graph->DeactivateSubgraph(subgraph);

  // Insert new node.
  for (const auto& op : reduce_ops) {
    graph->InsertNode(op);
  }
  graph->InsertNode(concat_op);
  LOG_DEBUG("concat_axis=%d reduce_axis=%d concat_input_size=%d", concat_axis, reduce_axis, concat_input_size);
  return true;
}

bool ConcatReduceSwapPass::IsReduceOp(const Node& child) {
  if (child.op.type() == "ReduceSum") {
    return true;
  }
  return false;
}

}  // namespace blaze
