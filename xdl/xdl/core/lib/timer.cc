/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "xdl/core/lib/timer.h"

#include <iomanip>
#include <iostream>

#include "xdl/core/utils/logging.h"

namespace xdl {

Timer::Timer(const char *name, TimerCore *tc) : name_(name), tc_(tc) {
}

Timer::~Timer() {
}

Timer &Timer::Display() {
  XDL_LOG(INFO) << std::setfill(' ') << std::setw(16) << name_  << " "
      << std::setw(8) << tc_->n << " "
      << std::setw(12) << std::setprecision(3) << ((float)tc_->du)/tc_->n/1e6 << " "
      << std::setw(12) << std::setprecision(3) << ((float)du_)/1e6;
  return *this;
}

Timer &Timer::Reset() {
  tc_->n = tc_->du = du_ = 0;
  return *this;
}

Timer &Timer::Step(int step) {
  step_ = step;
  return *this;
}

TimerScope::TimerScope(const char *name, TimerCore *tc) : Timer(name, tc) {
  Start();
}

TimerScope::~TimerScope() {
  Stop();
}


TimerCoreKeeper::TimerCoreKeeper() {
  memset(tc_vec_, 0, sizeof(tc_vec_));
}

TimerCoreKeeper::~TimerCoreKeeper() {
}

TimerCoreKeeper &TimerCoreKeeper::ResetAll() { memset(tc_vec_, 0, sizeof(tc_vec_[0]) * used_); }

TimerCore *TimerCoreKeeper::Get(const std::string &name) {
  std::lock_guard<std::mutex> lock { mtx_ };
  if (unlikely(used_ == capacity_)) {
    XDL_LOG(FATAL) << "Too many timers"; 
  }
  TimerCore *tc = &tc_vec_[used_++];
  auto ret = tc_map_.insert({name, tc});
  if (!ret.second) {
    --used_;
  }
  tc = ret.first->second;
  return tc;
}

TimerCoreKeeper &TimerCoreKeeper::Step(int step) {
  step_ = step;
  return *this;
}

}  // namespace xdl
