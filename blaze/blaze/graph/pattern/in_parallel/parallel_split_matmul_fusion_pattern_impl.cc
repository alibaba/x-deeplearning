/*
 * \file parallel_split_matmul_fusion_pattern_impl.cc 
 * \brief The parallel split matmul fusion pattern.
 * Such as: split->MatMul/MatMul/MatMul/...
 */
#include "blaze/graph/pattern/in_parallel/parallel_split_matmul_fusion_pattern_impl.h"

#include <sstream>

namespace blaze {

bool ParallelSplitMatMulFusionPatternImpl::BinaryNodeMatch(const ArgumentHelper* arg_node0,
                                                           const ArgumentHelper* arg_node1) {
  bool ltransA = arg_node0->GetSingleArgument<bool>("transA", false);
  bool ltransB = arg_node0->GetSingleArgument<bool>("transB", false);
  bool rtransA = arg_node1->GetSingleArgument<bool>("transA", false);
  bool rtransB = arg_node1->GetSingleArgument<bool>("transB", false);
  bool lfrom_deepnet = arg_node0->GetSingleArgument<bool>("from_deepnet", false);
  bool rfrom_deepnet = arg_node1->GetSingleArgument<bool>("from_deepnet", false);
  if (ltransA != rtransA || ltransB != rtransB ||
      lfrom_deepnet != rfrom_deepnet) return false;
  return true;
}

void ParallelSplitMatMulFusionPatternImpl::UpdateFusedBinaryArg(const ArgumentHelper* arg_node1,
                                                                OperatorDef* fused_op) {
  bool transA = arg_node1->GetSingleArgument<bool>("transA", false);
  bool transB = arg_node1->GetSingleArgument<bool>("transB", false);
  bool from_deepnet = arg_node1->GetSingleArgument<bool>("from_deepnet", false);

  Argument* arg = fused_op->add_arg();
  arg->set_name("transA");
  arg->set_i(transA);
  arg = fused_op->add_arg();
  arg->set_name("transB");
  arg->set_i(transB);
  arg = fused_op->add_arg();
  arg->set_name("from_deepnet");
  arg->set_i(from_deepnet);
}

}  // namespace blaze

