/*
 * \file parallel_split_reducesum_fusion_pattern_impl.cc 
 * \brief The parallel split reuducesum fusion pattern impl
 * Such as: Split->ReduceSum/ReduceSum/ReduceSum/... -> ReduceSum/Split
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_reducesum_fusion_pattern_impl.h"

#include "blaze/operator/common_helper.h"

namespace blaze {

bool ParallelSplitReduceSumFusionPatternImpl::UnaryNodeMatch(const ArgumentHelper* arg) {
  int axis = CommonHelper::GetReduceAxis(arg);
  if (axis <= 0) return false;
  return true;
}

bool ParallelSplitReduceSumFusionPatternImpl::UnaryNodeMatch(const ArgumentHelper* arg_node0,
                                                             const ArgumentHelper* arg_node1) {
  int axis0 = CommonHelper::GetReduceAxis(arg_node0);
  int axis1 = CommonHelper::GetReduceAxis(arg_node1);
  if (axis0 != axis1) return false;

  int keepdims0 = arg_node0->GetSingleArgument<int>("keepdims", 1);
  int keepdims1 = arg_node1->GetSingleArgument<int>("keepdims", 1);
  if (keepdims0 != keepdims1) return false;

  return true;
}

void ParallelSplitReduceSumFusionPatternImpl::UpdateFusedUnaryArg(const ArgumentHelper* arg_node1,
                                                                  OperatorDef* fused_op) {
  Argument* argument = fused_op->add_arg();
  argument->set_name("axis");
  argument->set_i(CommonHelper::GetReduceAxis(arg_node1));
  
  argument = fused_op->add_arg();
  argument->set_name("keepdims");
  argument->set_i(arg_node1->GetSingleArgument<int>("keepdims", 1));
}

}  // namespace blaze

