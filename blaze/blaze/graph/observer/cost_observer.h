/*
 * \file cost_observer.h
 * \beief The cost observer
 */
#pragma once

#include <sstream>

#include "blaze/common/timer.h"
#include "blaze/graph/observer/net_observer.h"
#include "blaze/operator/operator_schema.h"

namespace blaze {

class CostObserver;

class CostOperatorObserver : public ObserverBase<OperatorBase> {
 public:
  explicit CostOperatorObserver(OperatorBase* op) = delete;
  explicit CostOperatorObserver(OperatorBase* op, CostObserver* cost_observer) :
      ObserverBase<OperatorBase>(op), cost_observer_(cost_observer) { }

 protected:
  void Start() override { };
  void Stop() override;
  void Dump(std::string* out) override { };
  const char* Name() const override { return "cost_operator"; }

  OpSchema::Cost cost_;
  CostObserver* cost_observer_;
  friend class CostObserver;
};

class CostObserver : public NetObserver<CostOperatorObserver, CostObserver> {
 public:
  explicit CostObserver(Net* net) :
      NetObserver<CostOperatorObserver, CostObserver>(net, this) { }

  void Dump(std::string* out);
  const char* Name() const override { return "cost"; }

 protected:
  void Start() override {
    start_time_ = GetTime();
    cost_.Clear();
  }
  void Stop() override { end_time_ = GetTime(); }

  double start_time_, end_time_;
  OpSchema::Cost cost_;
  friend class CostOperatorObserver;
};

}  // namespace blaze
