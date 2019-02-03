/*!
 * \file concat_reduce_swap_pass.h
 * \brief The concat reduce swap 
 */
#pragma once

#include "blaze/optimizer/pass.h"

#include "blaze/graph/graph.h"

namespace blaze {

// Concat reduce swap for diminishing the memory copy overhead.
class ConcatReduceSwapPass : public Pass {
 public:
  ConcatReduceSwapPass& Name(std::string name);
  ConcatReduceSwapPass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);
  
 protected:
  bool ConcatReduceNodeSwapPass(Graph* graph, Node& node, void* arg);
  bool IsReduceOp(const Node& child);
};

}  // namespace blaze


