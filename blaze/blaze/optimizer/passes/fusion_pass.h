/*!
 * \file fusion_pass.h
 * \brief The fusion pass for kernel fusion.
 */
#pragma once

#include "blaze/optimizer/pass.h"

namespace blaze {

class FusionPass : public Pass {
 public:
  FusionPass& Name(std::string name);
  FusionPass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);
};

}  // namespace blaze


