/*
 * \file fused_parallel_mul_reduce_fusion_pattern_impl.h 
 * \brief The fused_parallel_mul and reduce fusion pattern
 */
#pragma once

#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class FusedParallelMulReduceFusionPatternImpl : public FusionPatternImpl {
 public:
  // The patten is matched.
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph rewrite
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;
};

}
