/*
 * \file profile_observer.h 
 * \brief The profile observer
 */
#pragma once

#include <sstream>

#include "blaze/graph/observer/net_observer.h"

namespace blaze {

class ProfileCounter {
 public:
  explicit ProfileCounter() { }

 protected:
  double start_time_ = 0.0f;
  double end_time_ = 0.0f;

#ifdef USE_CUDA
  bool init_ = false;
  cudaEvent_t start_;
  cudaEvent_t stop_;
#endif
};

// The net observer
class ProfileObserver;

class ProfileOperatorObserver : public ProfileCounter, public ObserverBase<OperatorBase> {
 public:
  explicit ProfileOperatorObserver(OperatorBase* op) = delete;
  explicit ProfileOperatorObserver(OperatorBase* op, ProfileObserver* profile_observer) :
      ObserverBase<OperatorBase>(op), profile_observer_(profile_observer) {}

 protected:
  void Start() override;
  void Stop() override;
  void Dump(std::string* out) override;
  const char* Name() const override { return "profile_operator"; }

  void CPUStart();
  void CUDAStart();

  void CPUStop();
  void CUDAStop();
  void CUDAStopFinal();

  ProfileObserver* profile_observer_;
  friend class ProfileObserver;
};

class ProfileObserver :
    public ProfileCounter,
    public NetObserver<ProfileOperatorObserver, ProfileObserver> {
 public:
  explicit ProfileObserver(Net* net) :
      NetObserver<ProfileOperatorObserver, ProfileObserver>(net, this) { }

  void Dump(std::string* out) override;
  const char* Name() const override { return "profile"; }

 protected:
  void Start() override;
  void Stop() override;

#ifdef USE_CUDA
  cudaEvent_t* begin_event_ = nullptr;
#endif

  size_t iterations_ = 0;
  double total_time_ = 0;
  size_t pid_ = 0;

  friend class ProfileOperatorObserver;
};

}  // namespace blaze

