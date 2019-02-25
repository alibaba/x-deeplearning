/*
 * \file dag_net.h 
 * \brief The dag net for topological sort parallel scheduling.
 * NOTE: Therse is no loops in dependency graph.
 * This DagNet is used for GPU scheduling.
 */
#pragma once

#include "blaze/graph/net.h"

#include <queue>
#include <list>

namespace blaze {

class DagNet : public Net {
 public:
  DagNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  // Get the operators.
  std::vector<OperatorBase*> GetOperators() override {
    std::vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

 protected:
  bool RunImpl() override;

  // Return the stream id.
  int AvailableStreamId(OperatorBase* op, int* stream_id);

  DISABLE_COPY_AND_ASSIGN(DagNet);
};

}  // namespace blaze

