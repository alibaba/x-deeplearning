/*
 * \file parallel_split_mul_fusion_pattern_impl.cc 
 * \brief The parallel split mul fusion pattern.
 * Such as: split->Mul/Mul/Mul/...
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_mul_fusion_pattern_impl.h"

#include <sstream>

namespace blaze {

bool ParallelSplitMulFusionPatternImpl::BinaryNodeMatch(const ArgumentHelper* arg_node0,
                                                        const ArgumentHelper* arg_node1) {
  return true;
}

void ParallelSplitMulFusionPatternImpl::UpdateFusedBinaryArg(const ArgumentHelper* arg_node1,
                                                             OperatorDef* fused_op) {
}

}  // namespace blaze

