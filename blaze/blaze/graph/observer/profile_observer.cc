/*
 * \file profile_observer.cc 
 * \brief The profile observer implementation 
 */
#include "blaze/graph/observer/profile_observer.h"

#include "blaze/common/timer.h"

namespace blaze {

void ProfileOperatorObserver::Start() {
  switch (profile_observer_->device_option().device_type()) {
    case kCPU:
      CPUStart();
      break;
    case kCUDA:
      CUDAStart();
      break;
  }
}

void ProfileOperatorObserver::CPUStart() {
  start_time_ = (GetTime() - profile_observer_->start_time_) * 1000 * 1000;
}

void ProfileOperatorObserver::CUDAStart() {
#ifdef USE_CUDA
  auto op = dynamic_cast<const Operator<CUDAContext>*>(subject_);
  if (op) {
    const auto& context = op->context();
    CUDADeviceGuard(context.device_id());
    if (!init_) {
      CUDA_CHECK(cudaEventCreate(&start_));
      CUDA_CHECK(cudaEventCreate(&stop_));
      init_ = true;
    }
    if (profile_observer_->begin_event_ == nullptr) {
      profile_observer_->begin_event_ = &start_;
    }
    CUDA_CHECK(cudaEventRecord(start_, context.cuda_stream()));
  } else {
    BLAZE_CONDITION_THROW("op is not on cuda");
  }
#endif
}

void ProfileOperatorObserver::Stop() {
  switch (profile_observer_->device_option().device_type()) {
    case kCPU:
      CPUStop();
      break;
    case kCUDA:
      CUDAStop();
      break;
  }
}

void ProfileOperatorObserver::CPUStop() {
  end_time_ = (GetTime() - profile_observer_->start_time_) * 1000 * 1000;
}

void ProfileOperatorObserver::CUDAStop() {
#ifdef USE_CUDA
  auto op = dynamic_cast<const Operator<CUDAContext>*>(subject_);
  if (op) {
    const auto& context = op->context();
    CUDADeviceGuard(context.device_id());
    CUDA_CHECK(cudaEventRecord(stop_, context.cuda_stream()));
  } else {
    BLAZE_CONDITION_THROW("op is not on cuda");
  }
#endif
}

void ProfileOperatorObserver::CUDAStopFinal() {
#ifdef USE_CUDA
  auto op = dynamic_cast<const Operator<CUDAContext>*>(subject_);
  if (op) {
    const auto& context = op->context();
    CUDADeviceGuard(context.device_id());
    
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float time = 0.0f;
    cudaEvent_t* se = profile_observer_->begin_event_;
    CUDA_CHECK(cudaEventElapsedTime(&time, *se, start_));
    start_time_ = time * 1000;
    CUDA_CHECK(cudaEventElapsedTime(&time, *se, stop_));
    end_time_ = time * 1000;
  }
#endif
}

void ProfileOperatorObserver::Dump(std::string* out) {
  const std::string& name = subject_->name();
  const std::string& type = subject_->type();

  std::stringstream ss;
  ss << "\t{\n\t\t\"cat\": \"" << type << "\"";
  ss << ",\n\t\t\"pid\": " << profile_observer_->pid_;
  ss << ",\n\t\t\"tid\": " << subject_->stream_id();
  ss << ",\n\t\t\"ts\": " << this->start_time_;
  ss << ",\n\t\t\"ph\": " << "\"B\"";
  ss << ",\n\t\t\"name\" : \"" << name << "\"";
  ss << "\n\t},\n";

  ss << "\t{\n\t\t\"cat\": \"" << type << "\"";
  ss << ",\n\t\t\"pid\": " << profile_observer_->pid_;
  ss << ",\n\t\t\"tid\": " << subject_->stream_id();
  ss << ",\n\t\t\"ts\": " << this->end_time_;
  ss << ",\n\t\t\"ph\": " << "\"E\"";
  ss << ",\n\t\t\"name\" : \"" << name << "\"";
  ss << "\n\t}";

  *out = ss.str();
}

void ProfileObserver::Start() {
  ++pid_;
  ++iterations_;
  start_time_ = GetTime();
  if (device_option().device_type() == kCUDA) {
#ifdef USE_CUDA
    begin_event_ = nullptr;
#endif
  }
}

void ProfileObserver::Stop() {
  total_time_ += GetTime() - start_time_;
  if (device_option().device_type() == kCUDA) {
#ifdef USE_CUDA
    for (auto operator_observer : operator_observers_) {
      operator_observer->CUDAStopFinal();
    }
#endif
  }
}

void ProfileObserver::Dump(std::string* out) {
  // write profile json files.
  std::stringstream ss;
  ss << "[\n";
  for (size_t k = 0; k < operator_observers_.size(); ++k) {
    if (k != 0) ss << ",\n";
    std::string str;
    operator_observers_[k]->Dump(&str);
    ss << str;
  }
  ss << "\n]";
  *out = ss.str();

#ifndef NPROFILE_EXPORT
  LOG_INFO("avrage total_time=%f op_num=%u",
           total_time_ / iterations_, operator_observers_.size());
  // NOTE: Just for testing only, later should be optimized.
  std::stringstream filename;
  filename << "profile/" << pid_ << ".json";
  FILE* fp = fopen(filename.str().c_str(), "w");
  if (fp == nullptr) {
    LOG_ERROR("open file %s failed", filename.str().c_str());
    return;
  }
  fwrite(ss.str().c_str(), 1, ss.str().length(), fp);
  fclose(fp);
#endif
}

}  // namespace blaze

