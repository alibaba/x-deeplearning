/*
 * \file gemm_gemm_fusion_pattern_impl.h 
 * \brief The gemm gemm fusion pattern.
 */
#pragma once

#include "blaze/common/blob.h"
#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class GemmGemmFusionPatternImpl : public FusionPatternImpl {
 public:
  // The patten is matched
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph reweite
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;

 protected:
  void CalcNewWeight(OperatorDef* w0,
                     bool transpose_w0,
                     OperatorDef* w1,
                     bool transpose_w1,
                     Blob* new_weight_blob);
  
  void CalcNewBias(float beta0,
                   OperatorDef* bias0,
                   OperatorDef* w1,
                   bool transpose_w1,
                   float beta1,
                   OperatorDef* bias1,
                   Blob* new_bias_blob,
                   int* bias_dtype);
};

}  // namespace blaze
