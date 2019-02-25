/*!
 * \file constant_pre_compute_pass.cc
 * \brief The constant pre compute pass
 */
#include "blaze/optimizer/passes/constant_pre_compute_pass.h"

#include "blaze/common/proto_helper.h"
#include "blaze/operator/common_helper.h"
#include "blaze/math/vml.h"

namespace blaze {

ConstantPreComputePass& ConstantPreComputePass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

ConstantPreComputePass& ConstantPreComputePass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef ConstantPreComputePass::RunPass(const NetDef& net_def) {
  // eliminate Reshape/ etc.
  Graph graph(net_def);
  Graph* graph_ptr = &graph;
  graph.BFS([this, graph_ptr](Node& node, void* arg) {
    return this->NodeConstantPreComputePass(graph_ptr, node, arg);
  }, nullptr);
  return graph.GetNetDef();
}

bool ConstantPreComputePass::NodeConstantPreComputePass(Graph* graph, Node& node, void* arg) {
  if (node.op.type() == "BatchNormalization") {
    return BatchNormalizationConstantPreComputePass(graph, node, arg);
  } else if (node.op.type() == "Dice") {
    return DiceConstantPreComputePass(graph, node, arg);
  }
  return false;
}

bool ConstantPreComputePass::BatchNormalizationConstantPreComputePass(Graph* graph, Node& node, void* arg) {
  ArgumentHelper node_argument_helper(node.op);
  if (node_argument_helper.GetSingleArgument<bool>("nosqrt", false)) {
    // Is already finish ConstantPreCompute
    return false;
  }
  float eps = node_argument_helper.GetSingleArgument<float>("eps", kBnEpsilon);

  // X, gamma(scale), beta(bias), mean, var
  // The four-the param.
  int parent_idx = node.GetParentIdx(node.op.input(4));   // Get var ConstantFill node
  Node& var_node = graph->node(parent_idx);

  ArgumentHelper argument_helper(var_node.op);
  int dtype = argument_helper.GetSingleArgument<int>("dtype", kFloat);
  TYPE_SWITCH(DataType2PassDataType(dtype), DType, {
    std::vector<DType> var = argument_helper.GetRepeatedArgument<DType>("value");
    for (size_t i = 0; i < var.size(); ++i) {
      var[i] += eps;
    }
    VML_Sqrt<DType, CPUContext>(var.size(), var.data(), const_cast<DType*>(var.data()), nullptr);
    ArgumentHelper::SetRepeatedArgument(var_node.op, "value", var);
    // set with no sqrt
    Argument* nosqrt = node.op.add_arg();
    nosqrt->set_name("nosqrt");
    nosqrt->set_i(1);
  });
  return false;
}

bool ConstantPreComputePass::DiceConstantPreComputePass(Graph* graph, Node& node, void* arg) {
  ArgumentHelper node_argument_helper(node.op);
  if (node_argument_helper.GetSingleArgument<bool>("nosqrt", false)) {
    // Is already finish ConstantPreCompute
    return false;
  }
  // X, gamma(scale), mean, var
  int parent_idx = node.GetParentIdx(node.op.input(3));
  Node& var_node = graph->node(parent_idx);

  ArgumentHelper argument_helper(var_node.op);
  int dtype = argument_helper.GetSingleArgument<int>("dtype", kFloat);
  TYPE_SWITCH(DataType2PassDataType(dtype), DType, {
    std::vector<DType> var = argument_helper.GetRepeatedArgument<DType>("value");
    for (size_t i = 0; i < var.size(); ++i) {
      var[i] += kDiceEpsilon; 
    }
    VML_Sqrt<DType, CPUContext>(var.size(), var.data(), const_cast<DType*>(var.data()), nullptr);
    ArgumentHelper::SetRepeatedArgument(var_node.op, "value", var);
    // set with no sqrt
    Argument* nosqrt = node.op.add_arg();
    nosqrt->set_name("nosqrt");
    nosqrt->set_i(1);
  });
  return false;
}

}  // namespace blaze


