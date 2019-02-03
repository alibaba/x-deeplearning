/*
 * \file net.h 
 * \brief Wrap operators togather with operator context.
 */
#pragma once

#include <functional>
#include <memory>

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/observer.h"
#include "blaze/common/log.h"
#include "blaze/graph/graph.h"
#include "blaze/graph/workspace.h"
#include "blaze/operator/operator.h"
#include "blaze/common/func.h"

namespace blaze {

class Net : public Observable<Net> {
 public:
  Net(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  virtual ~Net() noexcept { }

  inline const std::vector<const Event*>& events() const { return events_; }
  virtual void Wait() {
    for (const auto& event : events_) {
      event->Finish();
    }
  }

  virtual bool Run() {
    for (auto& op : GetOperators()) {
      op->ResetEvent();
    }

    StartAllObservers();
    if (!RunImpl()) {
      LOG_ERROR("Failed to run impl");
      return false;
    }
    Wait();  // Wait the net compuatation to be finished.
    
    StopAllObservers();
    HandleRunError();  

    return true;
  }

  virtual bool Run(const PredictorCallback&& cb) {
    auto success = Run();
    if (!success) return false;
    if (cb) cb();
    return true;
  }

  virtual void HandleRunError() {
    for (const Event* event : events_) {
      if (event->Query() != EventStatus::kEventSuccess) {
        LOG_ERROR("Error=%s", event->ErrorMessage().c_str());
      }
    }
  }

  inline const std::vector<std::string>& external_output() const {
    return external_output_;
  }
  inline const BlazeMap<std::string, Blob*>&  external_output_blob() const {
    return external_output_blob_;
  }
  inline Blob* external_output_blob(const std::string& name) {
    auto iter = external_output_blob_.find(name);
    if (iter == external_output_blob_.end()) return nullptr;
    else return iter->second;
  }

  inline const ValueInfo& external_input_info(const std::string& name) const {
    return workspace_->input_info(name);
  }
  inline bool has_external_input_info(const std::string& name) const {
    return workspace_->has_input_info(name); 
  }

  inline const std::vector<std::string>& external_input() const {
    return external_input_;
  }
  inline const BlazeMap<std::string, Blob*>& external_input_blob() const {
    return external_input_blob_;
  }
  inline Blob* external_input_blob(const std::string& name) const {
    const auto& iter = external_input_blob_.find(name);
    if (iter == external_input_blob_.end()) return nullptr;
    else return iter->second;
  }
  inline Blob* net_blob(const std::string& name) {
    auto iter = net_blob_map_.find(name);
    if (iter == net_blob_map_.end()) return nullptr;
    else return iter->second;
  }
  inline const BlazeMap<std::string, Blob*>& net_blob_map() const { return net_blob_map_; }
  virtual std::vector<std::string> GetTopoBlobName() const;

  // Used to attach Observers to operators of a Net
  virtual std::vector<OperatorBase*> GetOperators() = 0;

  const std::string& name() const { return name_; }
  std::string DebugStr();

  // Return the device option
  const DeviceOption& device_option() const { return net_def_->device_option(); }
  const NetDef& net_def() const { return *(net_def_.get()); }

  // Register all the observers
  void RegisterObservers();
  void RegisterObservers(const std::vector<std::string>& oberver_names);

  // Return the operators
  std::vector<std::unique_ptr<OperatorBase>>& operators() { return operators_; }

 protected:
  virtual bool RunImpl() {
    LOG_ERROR("Not implemented!");
    return false;
  }

  void CheckInputOutputConsistency();

  bool is_external_output(const std::string& name) {
    for (const auto& eo : external_output_) {
      if (eo == name) return true;
    }
    return false;
  }

  // If the op contains external output.
  void set_output_blob(const OperatorDef& operator_def, std::unique_ptr<OperatorBase>& op) {
    // Set the output
    for (size_t i = 0; i < operator_def.output_size(); ++i) {
      const std::string& output_str = operator_def.output(i);
      if (is_external_output(output_str)) {
        external_output_blob_[output_str] = op->Output(i);
      }
      net_blob_map_[output_str] = op->Output(i);
    }
  }

  bool is_external_input(const std::string& name) {
    for (const auto& ei : external_input_) {
      if (ei == name) return true;
    }
    return false;
  }

  // If the op contains external input.
  void set_input_blob(const OperatorDef& operator_def, std::unique_ptr<OperatorBase>& op) {
    // set the input
    for (size_t i = 0; i < operator_def.input_size(); ++i) {
      const std::string& input_str = operator_def.input(i);
      if (is_external_input(input_str)) {
        external_input_blob_[input_str] = op->Input(i);
      }
      net_blob_map_[input_str] = op->Input(i);
    }
  }

  std::vector<std::string> external_input_;
  std::vector<std::string> external_output_;
  std::string name_;
  std::vector<const Event*> events_;
  std::shared_ptr<const NetDef> net_def_;
  std::shared_ptr<Graph> graph_;
  BlazeMap<std::string, Blob*> external_output_blob_;
  BlazeMap<std::string, Blob*> external_input_blob_;
  // The total blob map.
  BlazeMap<std::string, Blob*> net_blob_map_;

  std::vector<std::unique_ptr<OperatorBase>> operators_;
  Workspace* workspace_;

  DISABLE_COPY_AND_ASSIGN(Net);
};

DECLARE_REGISTRY(NetRegistry,
                 Net,
                 const std::shared_ptr<const NetDef>&,
                 Workspace*);

#define REGISTER_NET(name, ...) \
    REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)

}  // namespace blaze
