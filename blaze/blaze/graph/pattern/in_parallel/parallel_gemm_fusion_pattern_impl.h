/*
 * \file parallel_gemm_fusion_pattern_impl.h 
 * \brief The parallel gemm fusion pattern
 * Such as: op->GEMM/GEMM/GEMM/.../GEMM
 */
#pragma once

#include "blaze/common/blob.h"
#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class ParallelGemmFusionPatternImpl : public FusionPatternImpl {
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
  struct GemmNodeInfo {
    int weight_dtype;
    std::vector<TIndex> shape;
    int bias_dtype;
    std::vector<TIndex> bias_shape;
    int idx;
    bool transA;
    bool transB;
    float alpha;
    float beta;
  };
  // Generate fusion candidate
  void GenerateFusionCandidate();
  // GemmNode[i] and GemmNode[j] can fusion?
  bool CanFusion(int i, int j);

  // Init fused gemm node, op params
  void InitFusedGemmOp(OperatorDef& op, const std::vector<int>& fusion_idx_sequence, Graph* graph);
  // Init fused split node, op params
  void InitFusedSplitOp(OperatorDef& op, const std::vector<int>& fusion_idx_sequence, Graph* graph);
  // Init fused Gemm Weight
  void InitFusedGemmWeight(const std::vector<int>& fusion_idx_sequence,
                           int fusion_weight_idx,
                           Graph* graph);
  // Init fused Gemm bias
  void InitFusedGemmBias(const std::vector<int>& fusion_idx_sequence,
                         int fusion_weight_idx,
                         Graph* graph);
  // Get gemm node info
  const GemmNodeInfo* GetGemmNodeInfo(int idx);

  std::vector<GemmNodeInfo> gemm_node_info_;
  std::vector<std::vector<int>> fusion_candidate_;
};

}  // namespace blaze
