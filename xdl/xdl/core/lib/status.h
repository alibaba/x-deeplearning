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

#ifndef XDL_CORE_LIB_STATUS_H_
#define XDL_CORE_LIB_STATUS_H_

#include <memory>
#include <string>
#include <algorithm>
#include <utility>

namespace xdl {

class Status {
 public:
  enum ErrorCode {
    kOk = 0,
    kArgumentError,
    kIndexOverflow,
    kInternal,
    kPsError,
    kOutOfRange,
    kReachEnd
  };

  Status() : state_(nullptr) {}
  ~Status() { delete state_; }

  Status(ErrorCode code, const std::string& msg)
    : state_(new State{code, msg}) {}

  Status(const Status& s)
      : state_(s.state_ == nullptr ? nullptr : new State(*s.state_)) {}
  void operator=(const Status& s) {
    if (!state_) delete state_;
    state_ = s.state_ == nullptr ? nullptr : new State(*s.state_);
  }

  Status(Status&& s) : state_(s.state_) { s.state_ = nullptr; }
  void operator=(Status&& s) { std::swap(state_, s.state_); }

  bool IsOk() const { return state_ == nullptr; }

  ErrorCode Code() const {
    return state_ == nullptr ? kOk : state_->code;
  }

  std::string Msg() const {
    return state_ == nullptr ? "" : state_->msg;
  }

  bool operator==(const Status& x) const {
    return (state_ == nullptr && x.state_ == nullptr) ||
        (state_ != nullptr && x.state_ != nullptr &&
         state_->code == x.state_->code && state_->msg == x.state_->msg);
  }
  bool operator!=(const Status& x) const {
    return !(*this == x);
  }

  std::string ToString() const {
    if (state_ == nullptr) {
      return "OK";
    } else {
      return "ErrorCode [" + std::to_string(Code()) + "]: " + Msg();
    }
  }

  static Status Ok() { return Status(); }
  static Status ArgumentError(const std::string& msg)
      { return Status(kArgumentError, msg); }
  static Status IndexOverflow(const std::string& msg)
      { return Status(kIndexOverflow, msg); }
  static Status Internal(const std::string& msg)
      { return Status(kInternal, msg); }
  static Status PsError(const std::string& msg)
      { return Status(kPsError, msg); }
  static Status OutOfRange(const std::string& msg)
      { return Status(kOutOfRange, msg); }
  static Status ReachEnd(const std::string& msg)
      { return Status(kReachEnd, msg); }

 private:
  struct State {
    ErrorCode code;
    std::string msg;
  };

  State* state_;
};

}  // namespace xdl

#define XDL_SINGLE_ARG(...) __VA_ARGS__

#define XDL_CHECK_STATUS(STATUS)                                    \
  do {                                                              \
    ::xdl::Status __st__ = STATUS;                                  \
    if (!__st__.IsOk()) {                                           \
      std::string msg = std::string("\nCheck Status [") + #STATUS   \
      + "] at [" __FILE__ + "]" + __func__ + "@"                    \
      + std::to_string(__LINE__);                                   \
      return ::xdl::Status(__st__.Code(), __st__.Msg() + msg);      \
    }                                                               \
  } while (0)

#define XDL_CHECK_STATUS_Q(STATUS)                                  \
  do {                                                              \
    xdl::Status __st__ = STATUS;                                    \
    if (!__st__.IsOk()) {                                           \
      return __st__;                                                \
    }                                                               \
  } while (0)

#define XDL_CHECK_STATUS_ASYNC(STATUS, CB)                          \
  do {                                                              \
    ::xdl::Status __st__ = STATUS;                                  \
    if (!__st__.IsOk()) {                                           \
      std::string msg = std::string("\nCheck Status [") + #STATUS   \
      + "] at [" __FILE__ + "]" + __func__ + "@"                    \
      + std::to_string(__LINE__);                                   \
      CB(::xdl::Status(__st__.Code(), __st__.Msg() + msg));         \
      return;                                                       \
    }                                                               \
  } while (0)

#define XDL_CHECK_STATUS_ASYNC_Q(STATUS, CB)                        \
  do {                                                              \
    xdl::Status __st__ = STATUS;                                    \
    if (!__st__.IsOk()) {                                           \
      CB(__st__);                                                   \
      return;                                                       \
    }                                                               \
  } while (0)

#define XDL_CHECK_COND(COND, STATUS)                                \
  do {                                                              \
    if (!(COND)) {                                                  \
      ::xdl::Status __st__ = STATUS;                                \
      std::string msg = std::string("\nCheck Condition [") + #COND  \
      + "] at [" __FILE__ + "]" + __func__ + "@"                    \
      + std::to_string(__LINE__);                                   \
      return ::xdl::Status(__st__.Code(), __st__.Msg() + msg);      \
    }                                                               \
  } while (0)

#define XDL_CHECK_COND_Q(STATUS)                                    \
  do {                                                              \
    if (!(COND)) {                                                  \
      return STATUS;                                                \
  } while (0)

#define XDL_CHECK_COND_ASYNC(COND, STATUS, CB)                      \
  do {                                                              \
    if (!(COND)) {                                                  \
      ::xdl::Status __st__ = STATUS;                                \
      std::string msg = std::string("\nCheck Condition [") + #COND  \
      + "] at [" __FILE__ + "]" + __func__ + "@"                    \
      + std::to_string(__LINE__);                                   \
      CB(::xdl::Status(__st__.Code(), __st__.Msg() + msg));         \
      return;                                                       \
    }                                                               \
  } while (0)

#define XDL_CHECK_COND_ASYNC_Q(STATUS, CB)                          \
  do {                                                              \
    if (!(COND)) {                                                  \
      CB(STATUS);                                                   \
      return;                                                       \
  } while (0)

#endif  // XDL_CORE_LIB_STATUS_H_
