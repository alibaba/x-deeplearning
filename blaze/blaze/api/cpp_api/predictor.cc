/*!
 * \file predictor.cc
 * \brief The blaze predictor cpp interface implementation.
 */
#include "blaze/api/cpp_api/predictor.h"

#include <string>
#include <mutex>

#include "blaze/api/cpp_api/predictor_impl.h"
#include "blaze/api/cpp_api/predictor_manager_impl.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/string_util.h"
#include "blaze/operator/common_helper.h"
#include "blaze/optimizer/optimizer.h"
#include "blaze/scheduler/scheduler_manager.h"

namespace blaze {

// Predictor Implementation
Predictor::Predictor(PredictorImpl* impl) {
  this->impl_ = impl;
}

Predictor::~Predictor() {
  delete this->impl_;
  this->impl_ = nullptr;
}

FeedNameConfig Predictor::GetFeedNameConfig(const std::string& feed_name) {
  return this->impl_->GetFeedNameConfig(feed_name);
}

bool Predictor::ReshapeInput(const char* name, const std::vector<size_t>& shape) {
  return this->impl_->ReshapeInput(name, shape);
}

bool Predictor::ReshapeInput(size_t idx, const std::vector<size_t>& shape) {
  return this->impl_->ReshapeInput(idx, shape);
}

bool Predictor::Feed(const char* name, const void* data, size_t len) {
  return this->impl_->Feed(name, data, len);
}

bool Predictor::Feed(size_t idx, const void* data, size_t len) {
  return this->impl_->Feed(idx, data, len);
}

PredictDataType Predictor::InputDataType(const char* name) const {
  return this->impl_->InputDataType(name);
}

PredictDataType Predictor::InputDataType(size_t idx) const {
  return this->impl_->InputDataType(idx);
}

size_t Predictor::InputSize() const {
  return this->impl_->InputSize();
}

const std::vector<std::string>& Predictor::ListInputName() const {
  return this->impl_->ListInputName();
}

bool Predictor::Forward(const PredictorCallback&& cb) {
  if (nullptr == cb) {
    return this->impl_->Forward(nullptr);
  } else {
    return this->impl_->Forward(std::move(cb));
  }
}

bool Predictor::Output(const char* name, void** data, size_t* len) {
  return this->impl_->Output(name, data, len);
}

bool Predictor::Output(size_t idx, void** data, size_t* size) {
  return this->impl_->Output(idx, data, size);
}

PredictDataType Predictor::OutputDataType(size_t idx) const {
  return this->impl_->OutputDataType(idx);
}

PredictDataType Predictor::OutputDataType(const char* name) const {
  return this->impl_->OutputDataType(name);
}

size_t Predictor::OutputSize() const {
  return this->impl_->OutputSize();
}

const std::vector<size_t>& Predictor::OutputShape(size_t idx) const {
  return this->impl_->OutputShape(idx);
}

const std::vector<size_t>& Predictor::OutputShape(const char* name) const {
  return this->impl_->OutputShape(name);
}

const std::vector<std::string>& Predictor::ListOutputName() const {
  return this->impl_->ListOutputName();
}

std::vector<std::string> Predictor::ListInternalName() const {
  return this->impl_->ListInternalName();
}

bool Predictor::InternalParam(const char* name, void** data, size_t* len) {
  return this->impl_->InternalParam(name, data, len);
}

const std::vector<size_t>& Predictor::InternalShape(const char* name) const {
  return this->impl_->InternalShape(name);
}

int Predictor::InternalDataType(const char* name) const {
  return this->impl_->InternalDataType(name);
}

void Predictor::RegisterObservers(const std::vector<std::string>& observer_names) {
  return this->impl_->RegisterObservers(observer_names);
}

void Predictor::DumpObservers(std::unordered_map<std::string, std::string>* dump_map) {
  return this->impl_->DumpObservers(dump_map);
}

PredictorManager::PredictorManager() {
  this->impl_ = new PredictorManagerImpl();
  LOG_INFO("New PredictorManager");
}

PredictorManager::~PredictorManager() {
  delete this->impl_;
  this->impl_ = nullptr;
}

void PredictorManager::SetDataType(PredictDataType data_type) {
  this->impl_->SetDataType(static_cast<DataType>(data_type));
}

void PredictorManager::SetRunMode(const char* run_mode) {
  this->impl_->SetRunMode(run_mode);
}

bool PredictorManager::LoadSparseModelWeight(const char* uri, const char* ps_puller_type) {
  return this->impl_->LoadSparseModelWeight(uri, ps_puller_type);
}

bool PredictorManager::LoadModel(const char* filename, bool optimization_pass) {
  return this->impl_->LoadModel(filename, "", kBlaze, optimization_pass);
}

bool PredictorManager::LoadDeepNetModel(const char* model_conf, const char* model_data, bool optimization_pass) {
  return this->impl_->LoadModel(model_conf, model_data, kUlf, optimization_pass);
}

bool PredictorManager::LoadModel(const char* conf_file, const char* data_file,
                                 ModelType model_type, bool optimization_pass) {
  return this->impl_->LoadModel(conf_file, data_file, model_type, optimization_pass);
}

Predictor* PredictorManager::CreatePredictor(PredictDeviceType predict_device, int device_id) {
  return this->impl_->CreatePredictor(predict_device, device_id);
}

std::string FeedNameUtility::SparseFeatureName2FeedName(const std::string& sparse_feature_name,
                                                        SparseFeatureType sft) {
  switch (sft) {
    case kSparseFeatureId:
      return sparse_feature_name + blaze::kIdSuffix;
    case kSparseFeatureValue:
      return sparse_feature_name + blaze::kValueSuffix;
    case kAuxSparseFeatureSegment:
      return sparse_feature_name + blaze::kIdNumSuffix; 
  }
}

std::string FeedNameUtility::IndicatorLevel2FeedName(int level) {
  return blaze::kIndicatorPrefix + blaze::kSparseFeatureSep + std::to_string(level);
}

bool InitScheduler(bool enable_batching,
                   int max_batch_size,
                   int batch_timeout_micros,
                   int num_threads_for_cpu,
                   int num_threads_for_cuda,
                   int num_threads_for_pipe) {
  auto scheduler_manager = SchedulerManager<AsyncTask>::Instance();
  SchedulerManager<AsyncTask>::Options options;
  options.enable_batching = enable_batching;
  options.max_batch_size = max_batch_size;
  options.batch_timeout_micros = batch_timeout_micros;
  options.num_threads_for_cpu = num_threads_for_cpu;
  options.num_threads_for_cuda = num_threads_for_cuda;
  options.num_threads_for_pipe = num_threads_for_pipe;

  return scheduler_manager->Init(options);
}

}  // namespace blaze
