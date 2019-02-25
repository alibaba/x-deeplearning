/*
 * \file simple_net.h 
 * \brief The simple net for sequential execution. 
 */
#pragma once

#include "blaze/graph/net.h"

namespace blaze {

class SimpleNet : public Net {
 public:
  SimpleNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  std::vector<OperatorBase*> GetOperators() override {
    std::vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

 protected:
  bool RunImpl() override; 

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

}  // namespace blaze

