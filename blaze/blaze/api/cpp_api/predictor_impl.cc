/*!
 * \file predictor_impl.cc
 * \brief The blaze predictor interface impl
 */
#include "blaze/api/cpp_api/predictor_impl.h"

#include "blaze/common/exception.h"
#include "blaze/common/string_util.h"
#include "blaze/operator/common_helper.h"
#include "blaze/graph/net.h"

namespace blaze {

PredictorImpl::PredictorImpl(std::shared_ptr<Net>& net) : net_(net) {
  // init external output index and blob vector.
  external_output_blob_.clear();
  external_output_blob_index_.clear();
  for (const auto& external_output : net_->external_output()) {
    const BlazeMap<std::string, Blob*>& external_output_blob = net_->external_output_blob();
    const auto& iter = external_output_blob.find(external_output);
    BLAZE_CONDITION_THROW(iter != external_output_blob.end(), "external_output: ",
                          external_output, " is not in external_output_blob");

    external_output_blob_index_[external_output] = external_output_blob_.size();
    external_output_blob_.push_back(iter->second);

    // set device type.
    DeviceOption device_option;
    device_option.set_device_type(kCPU);
    external_output_blob_cpu_.push_back(std::shared_ptr<Blob>(new Blob(device_option)));
    external_output_blob_cpu_[external_output_blob_cpu_.size() - 1]->set_data_type(
        static_cast<DataType>(iter->second->data_type()));
  }

  // init external input index and blob vector
  external_input_blob_.clear();
  external_input_blob_index_.clear();
  for (const auto& external_input : net_->external_input()) {
    const BlazeMap<std::string, Blob*>& external_input_blob = net_->external_input_blob();
    const auto& iter = external_input_blob.find(external_input);
    BLAZE_CONDITION_THROW(iter != external_input_blob.end(), "external_input: ",
                          external_input, " is not in external_input_blob");

    external_input_blob_index_[external_input] = external_input_blob_.size();
    external_input_blob_.push_back(iter->second);
  }
}

FeedNameConfig PredictorImpl::GetFeedNameConfig(const std::string& feed_name) {
  const auto& value_info = net_->external_input_info(feed_name);
  FeedNameConfig ret;
  ret.feature_name = value_info.feature_name();
  switch (value_info.input_type()) {
    case kInputDense:
      ret.feature_type = kDenseFeature;
      ret.feature_name = value_info.name();
      break;
    case kInputSparseIds:
      ret.feature_type = kSparseFeature;
      ret.sparse_feature_type = kSparseFeatureId;
      break;
    case kInputSparseValues:
      ret.feature_type = kSparseFeature;
      ret.sparse_feature_type = kSparseFeatureValue;
      break;
    case kInputSparseSegments:
      ret.feature_type = kSparseFeature;
      ret.sparse_feature_type = kAuxSparseFeatureSegment;
      break;
    case kInputIndicator:
      ret.feature_type = kAuxIndicator;
      break;
  }
  ret.level = value_info.level();
  return ret;
}

bool PredictorImpl::ReshapeInput(const char* name, const std::vector<size_t>& shape) {
  int idx = InputName2Idx(name);
  if (idx < 0) {
    LOG_ERROR("The name: %s is not The input of net", name);
    return false;
  }
  return ReshapeInput(idx, shape);
}

bool PredictorImpl::ReshapeInput(size_t idx, const std::vector<size_t>& shape) {
  if (idx >= external_input_blob_.size()) {
    LOG_ERROR("The idx=%u exceed size=%u", idx, external_input_blob_.size());
    return false;
  }
  Blob* blob = external_input_blob_[idx];
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, idx=%u", idx);
    return false;
  }
  blob->Reshape(shape);
  return true;
}

bool PredictorImpl::Feed(const char* name, const void* data, size_t len) {
  int idx = InputName2Idx(name);
  if (idx < 0) {
    LOG_ERROR("The name: %s is not The input of net", name);
    return false;
  }
  return Feed(idx, data, len);
}

bool PredictorImpl::Feed(size_t idx, const void* data, size_t len) {
  if (idx >= external_input_blob_.size()) {
    LOG_ERROR("The idx=%u exceed size=%u", idx, external_input_blob_.size());
    return false;
  }
  Blob* blob = external_input_blob_[idx];
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, idx=%u", idx);
    return false;
  }
  if (len != blob->size() * DataTypeSize(blob->data_type())) {
    LOG_ERROR("The len=%u need size=%u blob->size()=%u",
              len, blob->size() * DataTypeSize(blob->data_type()),
              blob->size());
    return false;
  }
  const DeviceOption& device_option = blob->device_option();
  if (device_option.device_type() == kCUDA) {
#ifdef USE_CUDA
    CUDAContext context(device_option);
    CUDA_CHECK(cudaMemcpyAsync(blob->as<char>(), data, blob->size() * DataTypeSize(blob->data_type()),
                               cudaMemcpyHostToDevice, context.cuda_stream()));
    context.FinishDeviceComputation();
#endif
  } else {
    memcpy(blob->as<char>(), data, blob->size() * DataTypeSize(blob->data_type()));
  }
  return true;
}

size_t PredictorImpl::InputSize() const {
  return external_input_blob_.size();
}

PredictDataType PredictorImpl::InputDataType(const char* name) const {
  int idx = InputName2Idx(name);
  if (idx < 0) {
    LOG_ERROR("The name: %s is not The input of net", name);
    return kPDT_Invalid;
  }
  return InputDataType(idx);
}

PredictDataType PredictorImpl::InputDataType(size_t idx) const {
  if (idx >= external_input_blob_.size()) {
    LOG_ERROR("The idx=%u exceed size=%u", idx, external_input_blob_.size());
    return kPDT_Invalid;
  }
  Blob* blob = external_input_blob_[idx];
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, idx=%u", idx);
    return kPDT_Invalid;
  }
  if (blob->data_type() == kPDT_Float) return kPDT_Float;
  else if (blob->data_type() == kPDT_Float16) return kPDT_Float16;
  else return kPDT_Invalid;
}

const std::vector<std::string>& PredictorImpl::ListInputName() const {
  return net_->external_input();
}

bool PredictorImpl::Forward(const PredictorCallback&& cb) {
  try {
    if (nullptr == cb) {
      return net_->Run();
    } else {
      return net_->Run(std::move(cb));
    }
  } catch (std::exception& e) {
    LOG_ERROR("failed: %s", e.what());
    return false;
  }
}

bool PredictorImpl::Output(const char* name, void** data, size_t* len) {
  int idx = OutputName2Idx(name);
  if (idx < 0) {
    LOG_ERROR("The name: %s is not The output of net", name);
    return false;
  }
  return Output(idx, data, len);
}

bool PredictorImpl::Output(size_t idx, void** data, size_t* len) {
  if (idx >= external_output_blob_.size()) {
    LOG_ERROR("The idx=%u exceed size=%u", idx, external_output_blob_.size());
    return false;
  }
  Blob* blob = external_output_blob_[idx];
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, idx=%u", idx);
    return false;
  }
  std::shared_ptr<Blob>& cpu_blob = external_output_blob_cpu_[idx];
  cpu_blob->Reshape(blob->shape());

  const DeviceOption& device_option = blob->device_option();
  if (device_option.device_type() == kCUDA) {
#ifdef USE_CUDA
    CUDAContext context(device_option);
    CUDA_CHECK(cudaMemcpyAsync(cpu_blob->as<char>(),
                               blob->as<char>(),
                               blob->size() * DataTypeSize(blob->data_type()),
                               cudaMemcpyDeviceToHost,
                               context.cuda_stream()));
    context.FinishDeviceComputation();
#endif
    *data = cpu_blob->as<char>();
    if (len) *len = cpu_blob->size() * DataTypeSize(blob->data_type());
  } else {
    *data = blob->as<char>();
    if (len) *len = blob->size() * DataTypeSize(blob->data_type());
  }
  return true;
}

const std::vector<size_t>& PredictorImpl::OutputShape(size_t idx) const {
  Blob* blob = external_output_blob_[idx];
  BLAZE_CONDITION_THROW(blob != nullptr, "The idx=", idx, " blob is null");
  return blob->shape(); 
}

const std::vector<size_t>& PredictorImpl::OutputShape(const char* name) const {
  int idx = OutputName2Idx(name);
  BLAZE_CONDITION_THROW(idx >= 0, "The name ", name, " is not Output of net");
  return OutputShape(idx);
}

PredictDataType PredictorImpl::OutputDataType(size_t idx) const {
  if (idx >= external_output_blob_.size()) {
    LOG_ERROR("The idx=%u exceed size=%u", idx, external_output_blob_.size());
    return kPDT_Invalid;
  }
  Blob* blob = external_output_blob_[idx];
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, idx=%u", idx);
    return kPDT_Invalid;
  }
  if (blob->data_type() == kPDT_Float) return kPDT_Float;
  else if (blob->data_type() == kPDT_Float16) return kPDT_Float16;
  else return kPDT_Invalid;
}

PredictDataType PredictorImpl::OutputDataType(const char* name) const {
  int idx = OutputName2Idx(name);
  if (idx < 0) {
    LOG_ERROR("The name: %s is not The output of net", name);
    return kPDT_Invalid;
  } 
  return OutputDataType(idx);
}

size_t PredictorImpl::OutputSize() const {
  return external_output_blob_index_.size();
}

const std::vector<std::string>& PredictorImpl::ListOutputName() const {
  return net_->external_output();
}

std::vector<std::string> PredictorImpl::ListInternalName() const {
  return net_->GetTopoBlobName();
}

bool PredictorImpl::InternalParam(const char* name, void** data, size_t* len) {
  Blob* blob = net_->net_blob(name);
  if (blob == nullptr) {
    LOG_ERROR("The blob is null, name=%s", name);
    return false;
  }
  std::shared_ptr<Blob>& cpu_blob = external_output_blob_cpu_[0]; // Borrow output[0]'s memory
  cpu_blob->Reshape(blob->shape());

  const DeviceOption& device_option = blob->device_option();
  if (device_option.device_type() == kCUDA) {
#ifdef USE_CUDA
    CUDAContext context(device_option);
    CUDA_CHECK(cudaMemcpyAsync(cpu_blob->as<char>(),
                               blob->as<char>(),
                               blob->size() * DataTypeSize(blob->data_type()),
                               cudaMemcpyDeviceToHost,
                               context.cuda_stream()));
    context.FinishDeviceComputation();
#endif
    *data = cpu_blob->as<char>();
    if (len) *len = cpu_blob->size() * DataTypeSize(blob->data_type());
  } else {
    *data = blob->as<char>();
    if (len) *len = blob->size() * DataTypeSize(blob->data_type());
  }
  return true;
}

const std::vector<size_t>& PredictorImpl::InternalShape(const char* name) const {
  Blob* blob = net_->net_blob(name);
  BLAZE_CONDITION_THROW(blob != nullptr, "blob is nullptr");
  return blob->shape();
}

int PredictorImpl::InternalDataType(const char* name) const {
  Blob* blob = net_->net_blob(name);
  BLAZE_CONDITION_THROW(blob != nullptr, "blob is nullptr");
  return blob->data_type();
}

void PredictorImpl::RegisterObservers(const std::vector<std::string>& observer_names) {
  net_->RegisterObservers(observer_names);
}

void PredictorImpl::DumpObservers(std::unordered_map<std::string, std::string>* dump_map) {
  net_->Dump(*dump_map);
}

int PredictorImpl::InputName2Idx(const char* name) const {
  const auto& iter = external_input_blob_index_.find(name);
  if (iter == external_input_blob_index_.end()) return -1;
  return iter->second;
}

int PredictorImpl::OutputName2Idx(const char* name) const {
  const auto& iter = external_output_blob_index_.find(name);
  if (iter == external_output_blob_index_.end()) return -1;
  return iter->second;
}

}  // namespace blaze
