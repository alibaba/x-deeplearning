/*
 * \file simple_net.cc 
 * \brief The simple net for sequential execution. 
 */
#include "blaze/graph/simple_net.h"

namespace blaze {

SimpleNet::SimpleNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws) : Net(net_def, ws) {
  bool net_has_device_option = net_def->has_device_option();
  // initialize the operatos
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
  // set the wait event for the net, all the operators will be executed on
  // stream_id = 0, Just wait for the last operator will be OK.
  if (net_def->op_size()) {
    size_t idx = net_def->op_size() - 1;
    const Event* event = &(operators_[idx]->event());
    this->events_.push_back(event);
  }
}

bool SimpleNet::RunImpl() {
  for (auto& op : operators_) {
    LOG_DEBUG("run:%s %s", op->operator_def().type().c_str(),
              op->operator_def().name().c_str());
    bool res = op->Run();
    if (!res) {
      LOG_ERROR("Operator failed, name=%s type=%s", op->name().c_str(), op->type().c_str());
      return false;
    }
  }
  return true;
}

REGISTER_NET(simple, SimpleNet);

}  // namespace blaze
