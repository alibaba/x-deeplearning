/*
 * \file fused_parallel_mul_reduce_fusion_pattern_impl.cc 
 * \brief The fused_parallel_mul and reduce fusion pattern
 */
#include "blaze/graph/pattern/in_order/fused_parallel_mul_reduce_fusion_pattern_impl.h"

namespace blaze {

bool FusedParallelMulReduceFusionPatternImpl::Match(
    const std::vector<ArgumentHelper*>& args,
    const std::vector<Node*>& nodes,
    Graph* graph) {
  return true;
}

void FusedParallelMulReduceFusionPatternImpl::GraphRewrite(
    const std::vector<ArgumentHelper*>& args,
    std::vector<Node*>& nodes,
    Graph* graph) {
  int parallel_num = args[0]->GetSingleArgument<int>("parallel_num", 2);
  OperatorDef& op = nodes[nodes.size() - 1]->op;
  ArgumentHelper::SetSingleArgument<size_t>(op, "parallel_num", parallel_num);
}

}  // namespace blaze
