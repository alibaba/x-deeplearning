/*
 * \file parallel_split_mul_fusion_pattern_impl.h 
 * \brief The parallel split mul fusion pattern.
 * Such as: split->Mul/Mul/Mul/...
 */
#pragma once

#include "blaze/graph/pattern/in_parallel/parallel_split_binary_fusion_pattern_impl.h"

namespace blaze {

class ParallelSplitMulFusionPatternImpl : public ParallelSplitBinaryFusionPatternImpl {
 protected:
  // The binary operator node0/node1 can be fused?
  virtual bool BinaryNodeMatch(const ArgumentHelper* arg_node0,
                               const ArgumentHelper* arg_node1) override;
  // Update the fused binary op arg
  virtual void UpdateFusedBinaryArg(const ArgumentHelper* arg_node1, OperatorDef* fused_op) override;
};

}  // namespace blaze
