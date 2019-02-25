/*
 * \file store.cc
 * \brief The sparse puller
 */
#include "blaze/store/store.h"

#include "blaze/common/exception.h"

namespace blaze {
namespace store {

bool StoreCreationRegisterer::Register(const std::string& name, FCreateStore fcs) {
  const auto& iter = fcs_map_.find(name);
  if (iter != fcs_map_.end()) {
    BLAZE_THROW("StoreCreation name=", name, " is already registered");
  }
  fcs_map_[name] = fcs;
  return true;
}

Store* StoreCreationRegisterer::CreateStore(const std::string& name) {
  const auto& iter = fcs_map_.find(name);
  if (iter == fcs_map_.end()) return nullptr;
  return iter->second();
}

}  // namespace store
}  // namespace blaze
