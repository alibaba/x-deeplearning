/*
 * \file net_observer.h 
 * \brief The network observer. 
 */
#pragma once

#include <vector>

#include "blaze/common/common_defines.h"
#include "blaze/common/observer.h"
#include "blaze/graph/net.h"
#include "blaze/operator/operator.h"

namespace blaze {

template <typename TOpObserver, typename TNetObserver>
class NetObserver : public ObserverBase<Net> {
 public:
  explicit NetObserver(Net* net, TNetObserver* net_observer) : ObserverBase<Net>(net) {
    const auto& operators = net->GetOperators(); 
    for (auto* op : operators) {
      auto observer = blaze::make_unique<TOpObserver>(op, net_observer);
      auto* ob = observer.get();
      op->AttachObserver(std::move(observer));
      operator_observers_.push_back(ob);
    }
    device_option_ = net->device_option();
  }
  virtual ~NetObserver() { }

  // Return the device option
  const DeviceOption& device_option() const { return device_option_; }

 protected:
  std::vector<TOpObserver*> operator_observers_;
  DeviceOption device_option_;
};

}  // namespace blaze

