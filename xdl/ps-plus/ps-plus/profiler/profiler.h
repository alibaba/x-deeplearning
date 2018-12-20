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

#ifndef PROFILER_PROFILER_H_
#define PROFILER_PROFILER_H_

#include <sys/time.h>
#include <functional>
#include <unordered_map>
#include <memory>

namespace ps {
namespace profiler {

class Profiler {
 public:
  Profiler(size_t thread, size_t repeat, const std::string& name)
    : thread_(thread), repeat_(repeat), name_(name) {}

  static size_t now() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000ul + tv.tv_usec;
  }

  void Profile();

  void SetInit(std::function<void(size_t threads)> init) {
    init_ = init;
  }

  void SetTestCase(std::function<void(size_t thread_id, bool run)> testcase) {
    testcase_ = testcase;
  }

 private:
  std::function<void(size_t threads)> init_;
  std::function<void(size_t thread_id, bool run)> testcase_;
  size_t thread_;
  size_t repeat_;
  std::string name_;
};

struct ProfilerRegister {
 public:
  ProfilerRegister(size_t thread, size_t repeat, const std::string& name) {
    profiler_ = new Profiler(thread, repeat, name);
    Map()[name].reset(profiler_);
  }
  ProfilerRegister(const ProfilerRegister&) {
    // Just Ignore
  }
  ProfilerRegister& Init(std::function<void(size_t threads)> init) {
    profiler_->SetInit(init);
    return *this;
  };
  ProfilerRegister& TestCase(std::function<void(size_t thread_id, bool run)> testcase) {
    profiler_->SetTestCase(testcase);
    return *this;
  }
  static std::unordered_map<std::string, std::unique_ptr<Profiler>>& Map() {
    static std::unordered_map<std::string, std::unique_ptr<Profiler>> map;
    return map;
  }
 private:
  Profiler* profiler_;
};

}
}

#define PROFILE(NAME, MAX_THREAD, REPEAT) \
static auto PROFILER_REGISTER_##NAME = ps::profiler::ProfilerRegister(MAX_THREAD, REPEAT, #NAME)

#endif

