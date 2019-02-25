/*!
 * \file fusion_pass.cc
 * \brief The fusion pass for kernel fusion.
 */
#include "blaze/optimizer/passes/fusion_pass.h"

#include "blaze/graph/fusion_pattern.h"

namespace blaze {

FusionPass& FusionPass::Name(std::string name) {
  this->name_ = name;
  return *this;
}

FusionPass& FusionPass::Type(PassType pass_type) {
  this->pass_type_ = pass_type;
  return *this;
}

NetDef FusionPass::RunPass(const NetDef& net_def) {
  return FusionPatternPass(net_def);
}

}  // namespace blaze


