/*
 * \file concat_concat_fusion_pattern_impl.h 
 * \brief The concat concat fusion helper utility.
 */
#pragma once

#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class ConcatConcatFusionPatternImpl : public FusionPatternImpl {
 public:
  // The patten is matched.
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph rewrite.
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;
};

}  // namespace blaze

