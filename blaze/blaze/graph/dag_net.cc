/*
 * \file dag_net.cc 
 * \brief The dag net for topological sort parallel scheduling.
 */
#ifdef USE_CUDA

#include "blaze/graph/dag_net.h"

#include "blaze/graph/graph.h"

namespace blaze {

DagNet::DagNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws) :
    Net(net_def, ws) {
  bool net_has_device_option = net_def->has_device_option();
  // initialize the operators
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    std::unique_ptr<OperatorBase> op;
    if (!operator_def.has_device_option() && net_has_device_option) {
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      op = CreateOperator(temp_def, ws, idx);
    } else {
      op = CreateOperator(operator_def, ws, idx);
    }
    set_input_blob(operator_def, op);
    set_output_blob(operator_def, op);
    operators_.emplace_back(std::move(op));
  }
}

bool DagNet::RunImpl() {
  const std::vector<int> starts = graph_->not_dependency_idx();
  std::list<int> ready_list;
  std::vector<bool> visited(graph_->size(), false);
  std::vector<int> stream_id(graph_->size(), 0);

  // Concurrently schedule the tasks
  int prob_stream_id = -1;
  for (auto idx : starts) {
    OperatorBase* op = operators_[idx].get();
    stream_id[idx] = AvailableStreamId(op, &prob_stream_id);
    op->Run(stream_id[idx]);
    // Make The op's successor into ready task queue.
    const Node& node = this->graph_->node(idx);
    for (const auto& item : node.children) {
      if (!visited[item.first]) {
        visited[item.first] = true;
        ready_list.push_back(item.first);
      }
    }
  }
  
  // Check the ready queue.
  while (!ready_list.empty()) {
    for (auto it = ready_list.begin(); it != ready_list.end(); ) {
      // Check the dependency
      int idx = *it;
      const Node& node = this->graph_->node(idx);
      OperatorBase* op = operators_[idx].get();
      int remaining = 0, last_stream_id = 0;
      for (const auto& item : node.parents) {
        int parent_idx = item.first;
        OperatorBase* parent_op = operators_[parent_idx].get();
        if (parent_op->event().Query() != EventStatus::kEventSuccess) {
          remaining++;
          last_stream_id = stream_id[parent_idx];
        }
      }
      if (remaining > 0) {
        ++it;
      } else {
        // Start to schedule it.
        stream_id[idx] = AvailableStreamId(op, &prob_stream_id);
        op->Run(stream_id[idx]);
        for (const auto& item : node.children) {
          if (!visited[item.first]) {
            visited[item.first] = true;
            ready_list.push_back(item.first);
          }
        }
        it = ready_list.erase(it);
      }
    }
  }

  // Put the output node's event to the net's wait events.
  const std::vector<int>& final_node = this->graph_->not_be_dependent_idx();
  for (auto idx : final_node) {
    const Event* event = &(operators_[idx]->event());
    this->events_.push_back(event);
  }

  return true;
}

int DagNet::AvailableStreamId(OperatorBase* op, int* stream_id) {
  do {
    *stream_id = (*stream_id + 1) % kMaxStreamNum;
  } while (!op->StreamIsFree(*stream_id));
  return *stream_id;
}

REGISTER_NET(dag, DagNet);

}  // namespace blaze

#endif
