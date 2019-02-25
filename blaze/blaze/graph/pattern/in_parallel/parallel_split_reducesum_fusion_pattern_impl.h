/*
 * \file parallel_split_reducesum_fusion_pattern_impl.h 
 * \brief The parallel split reducesum fusion pattern.
 * Such as: Split->ReduceSum/ReduceSum/ReduceSum/... -> ReduceSum/Split
 */
#pragma once

#include "blaze/graph/pattern/in_parallel/parallel_split_unary_fusion_pattern_impl.h"

namespace blaze {

class ParallelSplitReduceSumFusionPatternImpl : public ParallelSplitUnaryFusionPatternImpl {
 protected:
  // The unary operator can be fused?
  virtual bool UnaryNodeMatch(const ArgumentHelper* arg) override;
  // The two unary operator can be fused?
  virtual bool UnaryNodeMatch(const ArgumentHelper* arg_node0, const ArgumentHelper* arg_node1) override;

  // Update the fused unary op arg
  virtual void UpdateFusedUnaryArg(const ArgumentHelper* arg_node1, OperatorDef* fused_op) override;
};

}  // namespace blaze
