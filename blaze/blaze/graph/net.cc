/*
 * \file net.cc 
 * \brief Wrap operators togather with operator context.
 */
#include "blaze/graph/net.h"

#include "blaze/graph/observer/profile_observer.h"
#include "blaze/graph/observer/cost_observer.h"

namespace blaze {

Net::Net(const std::shared_ptr<const NetDef>& def, Workspace* workspace) :
    name_(def->name()),
    net_def_(def),
    workspace_(workspace) {
  bool net_has_device_option = net_def_->has_device_option();
  // NOTE: The device must be set for net forward.
  BLAZE_CONDITION_THROW(net_has_device_option, "net has no device option");

  this->graph_.reset(new Graph(*(net_def_.get())));
  for (const auto& name : graph_->external_input()) {
    external_input_.push_back(name);
  }
  for (const auto& name : graph_->external_output()) {
    external_output_.push_back(name);
  }

  // Check the consistency. TODO: comment or not?
  // CheckInputOutputConsistency();

  // Go through the operators.
  std::set<std::string> known_tensors(external_input_.begin(), external_input_.end());
  std::set<std::string> remaining_output(external_output_.begin(), external_output_.end());

  for (const auto& tensor : known_tensors) {
    remaining_output.erase(tensor);
  }
  for (const OperatorDef& op : def->op()) {
    for (const std::string& in : op.input()) {
      if (!known_tensors.count(in)) {
        if (external_input_.size()) {
          BLAZE_THROW("op ",
                      op.type(),
                      ": Source for input",
                      in,
                      " is unkown for net ",
                      def->name(),
                      ", operator ",
                      op.DebugString());
        } else {
          LOG_INFO("op %s: input %s is unkown", op.type().c_str(), in.c_str());
        }
      }
    }
    for (const std::string& out : op.output()) {
      known_tensors.insert(out);
      remaining_output.erase(out);
    }
  }
}

void Net::CheckInputOutputConsistency() {
  bool consistency = true;
  for (const auto& input_name : external_input_) {
    size_t k = 0;
    LOG_DEBUG("Net def external input size:%u", net_def_->external_input_size());
    for (; k < net_def_->external_input_size(); ++k) {
      if (input_name == net_def_->external_input(k).name()) break;
    }
    if (k == net_def_->external_input_size()) {
      LOG_ERROR("input_name=%s not found in external_input of net_def", input_name.c_str());
      consistency = false;
      break;
    }
  }

  for (const auto& output_name : external_output_) {
    size_t k = 0;
    for (; k < net_def_->external_output_size(); ++k) {
      if (output_name == net_def_->external_output(k).name()) break;
    }
    if (k == net_def_->external_output_size()) {
      LOG_ERROR("output_name=%s not found is external_output of net_def", output_name.c_str());
      consistency = false;
      break;
    }
  }
  BLAZE_CONDITION_THROW(consistency, "CheckInputOutputConsistency Failed");
}
  
std::vector<std::string> Net::GetTopoBlobName() const {
  std::vector<std::string> name_vec;
  std::set<std::string> name_set;
  for (const auto& op : operators_) {
    const OperatorDef& def = op->operator_def();
    for (const auto& iname : def.input()) {
      if (name_set.count(iname) == 0) {
        name_vec.push_back(iname);
        name_set.insert(iname);
      }
    }
    for (const auto& oname : def.output()) {
      if (name_set.count(oname) == 0) {
        name_vec.push_back(oname);
        name_set.insert(oname);
      }
    }
  }
  return name_vec;
}

std::string Net::DebugStr() {
  return graph_->DebugStr();
}

void Net::RegisterObservers() {
  std::unique_ptr<ProfileObserver> profile_ob = blaze::make_unique<ProfileObserver>(this);
  this->AttachObserver(std::move(profile_ob));

  std::unique_ptr<CostObserver> cost_ob = blaze::make_unique<CostObserver>(this);
  this->AttachObserver(std::move(cost_ob));
}

void Net::RegisterObservers(const std::vector<std::string>& observer_names) {
  for (const auto& name : observer_names) {
    if (name == "profile") {
      std::unique_ptr<ProfileObserver> profile_ob = blaze::make_unique<ProfileObserver>(this);
      this->AttachObserver(std::move(profile_ob));
    } else if (name == "cost") {
      std::unique_ptr<CostObserver> cost_ob = blaze::make_unique<CostObserver>(this);
      this->AttachObserver(std::move(cost_ob));
    } else {
      LOG_ERROR("Unkown observer name: %s", name.c_str());
    }
  }
}

DEFINE_REGISTRY(NetRegistry,
                Net,
                const std::shared_ptr<const NetDef>&,
                Workspace*);

}  // namespace blaze

