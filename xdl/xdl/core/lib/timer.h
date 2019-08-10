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
#ifndef __XDL_CORE_LIB_TIMER_H__
#define __XDL_CORE_LIB_TIMER_H__

#include <string.h>
#include <sys/time.h>

#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>

#include "xdl/core/lib/singleton.h"

namespace xdl {

struct TimerCore {
  long du;
  long n;
};

class Timer {
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;
  typedef std::chrono::microseconds us;
  typedef std::chrono::milliseconds ms;

 public:
  explicit Timer(const char *name, TimerCore *tc);
  virtual ~Timer();

  inline void Start() {
    start_ = clock::now();
  }

  inline void Stop() {
    du_ = static_cast<long>(
        std::chrono::duration_cast<ns>(clock::now() - start_).count());

    tc_->n ++;
    tc_->du += du_;

    if (step_ != 0 && tc_->n % step_ == 0) {
      Display();
    }
  }


  Timer &Step(int step);
  Timer &Display();
  Timer &Reset();

 protected:
  const char *name_;
  TimerCore *tc_ = nullptr;

  std::chrono::time_point<clock> start_;
  long du_ = 0;

  int step_ = 1;
};

class TimerScope : public Timer {
 public:
  explicit TimerScope(const char *name, TimerCore *tc);
  virtual ~TimerScope();
};

class TimerCoreKeeper {
 public:
  TimerCoreKeeper();
  ~TimerCoreKeeper();

  TimerCore *Get(const std::string &name);
  TimerCoreKeeper &ResetAll();
  TimerCoreKeeper &Display(const char *name, const TimerCore *tc = nullptr);
  TimerCoreKeeper &DisplayAll();
  TimerCoreKeeper &Step(int step);

 private:
  static const int capacity_ = 1024;
  int used_ = 0;
  int step_ = 1;
  std::unordered_map<std::string, TimerCore *> tc_map_;
  TimerCore tc_vec_[capacity_];
  std::mutex mtx_;
};

typedef Singleton<TimerCoreKeeper> tc_keeper;

}  // namespace xdl


#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define XDL_TIMER_CORE(NAME)                            \
    ({                                                  \
     static xdl::TimerCore *tc##NAME = nullptr;         \
     if (unlikely(tc##NAME == nullptr)) {                     \
       tc##NAME = xdl::tc_keeper::Instance()->Get(#NAME);     \
     }                                                  \
     tc##NAME;                                                \
     })

/// INTERNAL, DONT USE FOLLOWING
#define XDL_TIMER_VAR(NAME)           XDL_TIMER_VAR0(NAME)
#define XDL_TIMER_VAR0(NAME)          NAME

#define XDL_TIMER_VAR_CAT(NAME, SEC)  XDL_TIMER_VAR_CAT0(NAME, SEC)
#define XDL_TIMER_VAR_CAT0(NAME, SEC) xdl_##NAME##SEC

#define XDL_TIMER_TAG_CAT(NAME, SEC)  XDL_TIMER_TAG(NAME) ":" XDL_TIMER_TAG(SEC)
#define XDL_TIMER_TAG(NAME)           XDL_TIMER_TAG0(NAME)
#define XDL_TIMER_TAG0(NAME)          #NAME


#ifndef NTIMER

/// KEEPER
#define XDL_TIMER_RESET_ALL()         xdl::tc_keeper::Instance()->ResetAll()


/// TIMER
#define XDL_TIMER(NAME)               static xdl::Timer NAME(XDL_TIMER_TAG(NAME), XDL_TIMER_CORE(NAME))

/// TIMER ACTION
#define XDL_TIMER_START(NAME)         NAME.Start()
#define XDL_TIMER_STOP(NAME)          NAME.Stop()
#define XDL_TIMER_RESET(NAME)         NAME.Reset()
#define XDL_TIMER_DISPLAY(NAME)       NAME.Display()

/// TIMER START NOW
#define XDL_TIMER_NOW(NAME)           XDL_TIMER(NAME); XDL_TIMER_START(NAME)

/// TIMER SCOPE
#define XDL_TIMER_SCOPE(NAME)         xdl::TimerScope NAME(XDL_TIMER_TAG(NAME), XDL_TIMER_CORE(NAME))

#else

#define XDL_TIMER_DISPLAY_ALL()
#define XDL_TIMER_RESET_ALL()
#define XDL_TIMER_SET_STEP(STEP)
#define XDL_TIMER_STEP

/// TIMER
#define XDL_TIMER(NAME)
                                       
/// TIMER START NOW
#define XDL_TIMER_NOW(NAME)

/// TIMER ACTION
#define XDL_TIMER_START(NAME)
#define XDL_TIMER_STOP(NAME)
#define XDL_TIMER_RESET(NAME)
#define XDL_TIMER_DISPLAY(NAME)
 
/// TIMER SCOPE
#define XDL_TIMER_SCOPE(NAME)

#endif  // NTIMER

#endif  // __XDL_CORE_LIB_TIMER_H__
