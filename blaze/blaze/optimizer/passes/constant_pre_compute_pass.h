/*!
 * \file constant_pre_compute_pass.h
 * \brief The pre compute part.
 */
#pragma once

#include "blaze/optimizer/pass.h"
#include "blaze/graph/graph.h"

namespace blaze {

class ConstantPreComputePass : public Pass {
 public:
  ConstantPreComputePass& Name(std::string name);
  ConstantPreComputePass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);

 protected:
  bool NodeConstantPreComputePass(Graph* graph, Node& node, void* arg);

  bool BatchNormalizationConstantPreComputePass(Graph* graph, Node& node, void* arg);
  bool DiceConstantPreComputePass(Graph* graph, Node& node, void* arg);
};

}  // namespace blaze
