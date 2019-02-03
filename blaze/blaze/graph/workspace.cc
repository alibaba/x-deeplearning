/*
 * \file workspace.cc
 * \desc The run-time workspace implementation
 */
#include "blaze/graph/workspace.h"

#include "blaze/common/proto_configure.h"
#include "blaze/graph/net.h"

namespace blaze {

void Workspace::Init(const NetDef& net_def) {
  std::unique_lock<std::mutex> lock(mutex_);
  scheduler_manager_ = SchedulerManager<AsyncTask>::Instance();
  
  Clear();

  net_def_.reset(new NetDef(net_def));
  // init data type map of input and output
  InitInputOutputDataTypeMap();
  // init info of input
  InitInputInfoMap();
}

std::shared_ptr<Net> Workspace::CreateNet() {
  std::unique_lock<std::mutex> lock(mutex_);
  RecycleBlobMap();
  std::string run_mode = net_def_->run_mode();
  std::shared_ptr<Net> net = NetRegistry()->Create(run_mode, net_def_, this);
  return net;
}

std::shared_ptr<Net> Workspace::CreateNet(const char* net_file) {
  try {
    ProtoConfigure proto_conf("blaze.NetDef", net_file);
    const NetDef* net_def = dynamic_cast<const NetDef*>(proto_conf.config());
    Init(*net_def);
  } catch (...) {
    BLAZE_THROW("Open net_file: ", net_file, " failed");
  }
  return CreateNet();
}

void Workspace::InitInputOutputDataTypeMap() {
  for (size_t k = 0; k < net_def_->external_input_size(); ++k) {
    const auto& input = net_def_->external_input(k);
    input_data_type_map_[input.name()] = input.dtype();
  }
  for (size_t k = 0; k < net_def_->external_output_size(); ++k) {
    const auto& output = net_def_->external_output(k);
    output_data_type_map_[output.name()] = output.dtype();
  }
}

void Workspace::InitInputInfoMap() {
  for (size_t k = 0; k < net_def_->external_input_size(); ++k) {
    const auto& input = net_def_->external_input(k);
    input_info_map_[input.name()] = input;
  }
}

void Workspace::RecycleBlobMap() {
  for (auto& item : blob_map_) {
    Blob* blob = item.second;
    if (blob_recycle_bins_.count(blob)) continue;
    blob_recycle_bins_.insert(blob);
  }
  blob_map_.clear();
}

}  // namespace blaze

