/*
 * \file parallel_split_softmax_fusion_pattern_impl.cc 
 * \brief The parallel split softmax fusion pattern impl
 * Such as: Split->Softmax/Softmax/Softmax/... -> Softmax/Split
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_softmax_fusion_pattern_impl.h"

#include <sstream>

namespace blaze {

bool ParallelSplitSoftmaxFusionPatternImpl::UnaryNodeMatch(const ArgumentHelper* arg) {
  if (arg->GetSingleArgument<int>("axis", 1) <= 0) return false;
  return true;
}

bool ParallelSplitSoftmaxFusionPatternImpl::UnaryNodeMatch(const ArgumentHelper* arg_node0,
                                                           const ArgumentHelper* arg_node1) {
  if (arg_node0->GetSingleArgument<int>("axis", 1) !=
      arg_node1->GetSingleArgument<int>("axis", 1)) return false;
  return true;
}

void ParallelSplitSoftmaxFusionPatternImpl::UpdateFusedUnaryArg(const ArgumentHelper* arg_node1,
                                                                OperatorDef* fused_op) {
  Argument* argument = fused_op->add_arg();
  argument->set_name("axis");
  argument->set_i(arg_node1->GetSingleArgument<int>("axis", 1));
}

}  // namespace blaze

