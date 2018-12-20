/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PS_PLUS_COMMON_METRICS_COLLECTOR_H_
#define PS_PLUS_COMMON_METRICS_COLLECTOR_H_

#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <iostream>

namespace ps {
namespace common {

struct Metrics {
  Metrics(size_t cur)
    : count_(0)
    , total_(0)
    , start_(cur)
    , max_(0) 
    , except_(0) {
  }
  Metrics() = default;
  size_t count_;
  size_t total_;
  size_t start_;
  size_t max_;
  size_t except_;
};

class MetricsCollector {
private:
  MetricsCollector() {}

public:
  static MetricsCollector* Instance() {
    static MetricsCollector mc;
    return &mc;
  }

  ~MetricsCollector() {}

  void Start() {
    th_ = std::thread([this]() {
	while (true) {
	  Print();
	  std::this_thread::sleep_for(std::chrono::seconds(30));	  
	}
      });
  }

  void Stop() {
    th_.join();
  }

  void Begin(const std::string& name) {
    std::lock_guard<std::mutex> l(mu_);
    if (metrics_.find(name) == metrics_.end()) {
      metrics_[name] = Metrics(GetCurrentTimeMS());
    } else {
      metrics_[name].start_ = GetCurrentTimeMS();
    }
  }
  
  void End(const std::string& name) {
    std::lock_guard<std::mutex> l(mu_);
    metrics_[name].count_++;
    size_t duration = GetCurrentTimeMS() - metrics_[name].start_;
    metrics_[name].total_ += duration;
    if (duration > metrics_[name].max_) {
      metrics_[name].max_ = duration;
    }

    if (duration > (metrics_[name].total_ / metrics_[name].count_) * 10) {
      metrics_[name].except_++;
    }
  }

  void RecordNow(const std::string&name, size_t begin) {
    Record(name, GetCurrentTimeMS() - begin);
  }

  void Record(const std::string& name, size_t duration) {
    std::lock_guard<std::mutex> l(mu_);    
    if (metrics_.find(name) == metrics_.end()) {
      metrics_[name] = Metrics(GetCurrentTimeMS());
    } 

    metrics_[name].count_++;
    metrics_[name].total_ += duration;
    if (duration > metrics_[name].max_) {
      metrics_[name].max_ = duration;
    }

    if (duration > 20000) {
      metrics_[name].except_++;
    }
  }

  void Print() {
    std::unordered_map<std::string, Metrics> tmp;    
    {
      std::lock_guard<std::mutex> l(mu_);
      tmp = metrics_;
      for (auto& item: metrics_) {
	item.second.max_ = 0;
      }
    }

    for (auto& item: tmp) {
      size_t avg = item.second.total_ / item.second.count_;
      if (avg >= 1000 && item.first.find("^") != 0) {
	std::cout << item.first << ":" << "count[" << 
	  item.second.count_ << "] avg[" << 
	  avg << "] max[" << 
	  item.second.max_ << "] except[" << item.second.except_ << "]" << 
	  std::endl;
      }
    }
  }

  static size_t GetCurrentTimeMS() {
    auto time_now = std::chrono::system_clock::now();  
    auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());  
    return duration_in_ms.count();
  }

private:
  std::unordered_map<std::string, Metrics> metrics_;
  std::mutex mu_;
  std::thread th_;
};

}
}

#endif
