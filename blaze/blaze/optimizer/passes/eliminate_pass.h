/*!
 * \file eliminate_pass.h
 * \brief The eliminate pass for eliminating useleass op.
 */
#pragma once

#include "blaze/optimizer/pass.h"
#include "blaze/graph/graph.h"

namespace blaze {

class EliminatePass : public Pass {
 public:
  EliminatePass& Name(std::string name);
  EliminatePass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);

 protected:
  bool NodeEliminatePass(Graph* graph, Node& node, void* arg);
  bool AddNodeEliminatePass(Graph* graph, Node& node, void* arg);
};

}  // namespace blaze


