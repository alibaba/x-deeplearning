/*
 * \file observer.h
 * \desc The observer pattern template
 */
#pragma once

#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace blaze {

template <class T>
class ObserverBase {
 public:
  explicit ObserverBase(T* subject) : subject_(subject) { }
  virtual ~ObserverBase() noexcept { }

  virtual void Start() { }
  virtual void Stop() { }
  virtual void Dump(std::string* out) { }

  virtual const char* Name() const { return "Not implemented!"; }
  T* subject() const { return subject_; }

 protected:
  T* subject_;
};

// Inherit to make your class observable
template <class T>
class Observable {
 public:
  virtual ~Observable() { }
  using Observer = ObserverBase<T>;
  
  // Return a reference to the observer after addition
  const Observer* AttachObserver(std::unique_ptr<Observer> observer) {
    std::unordered_set<const Observer*> observers;
    for (auto& ob : observers_list_) {
      observers.insert(ob.get());
    }
    const auto* observer_ptr = observer.get();
    if (observers.count(observer_ptr)) {
      return observer_ptr;
    }
    observers_list_.push_back(std::move(observer));
    return observer_ptr;
  }

  // Return a remove observer pointer
  std::unique_ptr<Observer> DetachObserver(const Observer* observer_ptr) {
    for (auto it = observers_list_.begin();
         it != observers_list_.end(); ++it) {
      if (it->get() == observer_ptr) {
        auto res = std::move(*it);
        observers_list_.erase(it);
        return res;
      }
    }
    return nullptr;
  }

  // Return number of observers
  virtual size_t NumOfObservers() { return observers_list_.size(); }

  // Start all the observers
  void StartAllObservers() {
    for (auto& observer : observers_list_) {
      observer->Start();
    }
  }
  // Stop all the observers
  void StopAllObservers() {
    for (auto& observer : observers_list_) {
      observer->Stop();
    }
  }
  void Dump(std::unordered_map<std::string, std::string>& dump_map) {
    for (auto& observer : observers_list_) {
      std::string out;
      observer->Dump(&out);
      dump_map[observer->Name()] = out;
    }
  }

 protected:
  std::vector<std::unique_ptr<Observer>> observers_list_;
};

}  // namespace blaze
