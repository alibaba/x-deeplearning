/*!
 * \file gemm_pass.cc
 * \brief The gemm pass for Gemm op's param reinit
 */
#include "blaze/optimizer/passes/gemm_pass.h"

#include "blaze/common/proto_helper.h"

namespace blaze {

GemmPass& GemmPass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

GemmPass& GemmPass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef GemmPass::RunPass(const NetDef& net_def) {
  bool conti = true;
  NetDef ret = net_def;
  while (conti) {
    Graph graph(ret);
    Graph* graph_ptr = &graph;
    conti = graph.BFS([this, graph_ptr](Node& node, void* arg) {
                if (node.op.type() == "Gemm") {
                  return this->RunGemmPass(graph_ptr, node, arg);
                } else {
                  return false;
                }
            }, nullptr);
    ret = graph.GetNetDef();
  }
  return ret;
}

bool GemmPass::RunGemmPass(Graph* graph, Node& node, void* arg) {
  BLAZE_CONDITION_THROW(node.op.input_size() >= 2, "node.op.input_size()=",
                        node.op.input_size());
  auto weight_idx = node.GetParentIdx(node.op.input(1));
  ProcessGemmParam(graph->node(weight_idx), 2);
  if (node.op.input_size() > 2) {
    auto bias_idx = node.GetParentIdx(node.op.input(2));
    ProcessGemmParam(graph->node(bias_idx), 1);
  }
  return false;
}

void GemmPass::ProcessGemmParam(Node& node, int min_size) {
  ArgumentHelper proto_helper(node.op);
  auto shape = proto_helper.GetRepeatedArgument<TIndex>("shape");

  // calculate prefix_index.
  int prefix_index = 0;
  for (; prefix_index + min_size < shape.size() && shape[prefix_index] == 1; ++prefix_index);

  if (prefix_index != 0) {
    for (auto i = 0; i < node.op.arg_size(); ++i) {
      if (node.op.arg(i).name() == "shape") {
        node.op.mutable_arg(i)->clear_ints();
        for (auto k = prefix_index; k < shape.size(); ++k) {
          node.op.mutable_arg(i)->add_ints(shape[k]);
        }
        break;
      }
    }
  }
}

}  // namespace blaze
