/*!
 * \file pass.h
 * \brief The pass base class
 */
#pragma once

#include "blaze/graph/graph.h"
#include "blaze/graph/workspace.h"

namespace blaze {

enum PassType {
  // The fusion ect. optimization based on graph, which can be
  // executed offline
  kGraph = 0,

  // The memory optimization, which will be optimized online.
  kWorkspace,
};

class Pass {
 public:
  virtual NetDef RunPass(const NetDef& net_def) { return net_def; }
  virtual NetDef RunPass(const NetDef& net_def, Workspace* ws) { return net_def; }

  PassType pass_type() const { return pass_type_; }
  const std::string& name() const { return name_; }

 protected:
  PassType pass_type_;
  std::string name_;
};

// Pass register
struct PassRegisterer {
  static PassRegisterer* Get() {
    static std::shared_ptr<PassRegisterer> inst(new PassRegisterer());
    return inst.get();
  }
  template <typename T>
  T& Register() {
    size_t idx = pass.size();
    pass.resize(idx + 1);
    pass[idx].reset(new T());
    return *(dynamic_cast<T*>(pass[idx].get()));
  }

  std::vector<std::shared_ptr<Pass>> pass;
};

#define REGISTER_PASS(PassClass)                                \
    static PassClass& ANONYMOUS_VARIABLE(PassClass) =           \
      PassRegisterer::Get()->Register<PassClass>()

}  // namespace blaze

