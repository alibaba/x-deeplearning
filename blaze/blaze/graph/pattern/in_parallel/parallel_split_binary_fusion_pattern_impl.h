/*
 * \file parallel_split_binary_fusion_pattern_impl.h 
 * \brief The parallel split binary fusion pattern.
 * Such as: split->Binary/Binary/Binary/...
 */
#pragma once

#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class ParallelSplitBinaryFusionPatternImpl : public FusionPatternImpl {
 public:
  virtual void Init() override;
  // The pattern is matched
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph rewrite
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;

 protected:
  // Find node from idxs, whose a_node is a_node_idx, b_node is b_node_idx,
  // and the input/output relationship is a_output_idx_cur/b_output_idx_cur.
  int FindMatchedNode(Graph* graph, const std::set<int>& ids,
                      int a_node_idx, int a_output_idx_cur,
                      int b_node_idx, int b_output_idx_cur);

  bool InitFusedSplitOp(OperatorDef& op, int node_idx,
                        int begin, int end, Graph* graph);
  void InitFusedBinaryOp(OperatorDef& op, OperatorDef& split_node,
                         const std::vector<int>& binary_idx,
                         const ArgumentHelper* arg_node1,
                         Graph* graph);

  // The binary operator node0/node1 can be fused?
  virtual bool BinaryNodeMatch(const ArgumentHelper* arg_node0,
                               const ArgumentHelper* arg_node1) = 0;
  // Update the fused binary op arg
  virtual void UpdateFusedBinaryArg(const ArgumentHelper* arg_node1, OperatorDef* fused_op) = 0;
  
  std::vector<int> binary_idx_;
  int a_node_idx_, b_node_idx_;
  // [a_node_output_begin_, b_node_output_end_)
  int a_node_output_begin_, a_node_output_end_;
  // [a_node_output_begin_, b_node_output_end_)
  int b_node_output_begin_, b_node_output_end_;
  bool transA_, transB_;
};

}  // namespace blaze
