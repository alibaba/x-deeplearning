/*
 * \file parallel_split_unary_fusion_pattern_impl.h 
 * \brief The parallel split unary fusion pattern
 * Such as: Split->Softmax/Softmax/Softmax/... -> Softmax/Split
 */
#pragma once

#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class ParallelSplitUnaryFusionPatternImpl : public FusionPatternImpl {
 public:
  virtual void Init() override;
  // The pattern is matched.
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph rewrite
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;

 protected:
  // The unary operator can be fused?
  virtual bool UnaryNodeMatch(const ArgumentHelper* arg) = 0;
  // The two unary operations can fusion?
  virtual bool UnaryNodeMatch(const ArgumentHelper* arg_node0, const ArgumentHelper* arg_node1) = 0;

  // Update the fused unary op arg
  virtual void UpdateFusedUnaryArg(const ArgumentHelper* arg_node1, OperatorDef* fused_op) = 0;

  std::vector<int> candidate_idx_;
  int split_node_idx_;
};

}  // namespace blaze
