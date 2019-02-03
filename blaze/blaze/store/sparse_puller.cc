/*
 * \file sparse_puller.cc 
 * \brief The sparse puller
 */
#include "blaze/store/sparse_puller.h"

#include "blaze/common/exception.h"

namespace blaze {
namespace store {

bool SparsePullerCreationRegisterer::Register(const std::string& name, FCreateSparsePuller fcs) {
  const auto& iter = fcs_map_.find(name);
  if (iter != fcs_map_.end()) {
    BLAZE_THROW("SparsePullerCreation name=", name, " is already registered");
  }
  fcs_map_[name] = fcs;
  return true; 
}

SparsePuller* SparsePullerCreationRegisterer::CreateSparsePuller(const std::string& name) {
  const auto& iter = fcs_map_.find(name);
  if (iter == fcs_map_.end()) return nullptr;
  return iter->second();
}

}  // namespace store
}  // namespace blaze
