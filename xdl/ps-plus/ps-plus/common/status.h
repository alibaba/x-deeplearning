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

#ifndef PS_COMMON_STATUS_H_
#define PS_COMMON_STATUS_H_

#include <memory>
#include <string>
#include <algorithm>

namespace ps {

namespace serializer {
class SerializeHelper;
}

class Status {
 public:
  enum ErrorCode {
    kOk = 0,
    kArgumentError,
    kIndexOverflow,
    kNotFound,
    kDataLoss,
    kAlreadyExist,
    kNotImplemented,
    kUdfNotRegistered,
    kVersionMismatch,
    kConcurrentExecution,
    kNotReady,
    kNetworkError,
    kTimeout,
    kServerFuncNotFound,
    kServerSerializeFailed,
    kServerDeserializeFailed,
    kClientSerializeFailed,
    kClientDeserializeFailed,
    kFileQueueNeedWait,
    kUnknown
  };

  Status() : state_(nullptr) {}
  ~Status() { delete state_; }

  Status(ErrorCode code, const std::string& msg)
    : state_(new State{code, msg}) {}

  Status(const Status& s) : state_(s.state_ == nullptr ? nullptr : new State(*s.state_)) {}
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
      (state_ != nullptr && x.state_ != nullptr && state_->code == x.state_->code && state_->msg == x.state_->msg);
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
  static Status ArgumentError(const std::string& msg) { return Status(kArgumentError, msg); }
  static Status IndexOverflow(const std::string& msg) { return Status(kIndexOverflow, msg); }
  static Status NotFound(const std::string& msg) { return Status(kNotFound, msg); }
  static Status DataLoss(const std::string& msg) { return Status(kDataLoss, msg); }
  static Status AlreadyExist(const std::string& msg) { return Status(kAlreadyExist, msg); }
  static Status NotImplemented(const std::string& msg) { return Status(kNotImplemented, msg); }
  static Status UdfNotRegistered(const std::string& msg) { return Status(kUdfNotRegistered, msg); }
  static Status VersionMismatch(const std::string& msg) { return Status(kVersionMismatch, msg); }
  static Status ConcurrentExecution(const std::string& msg) { return Status(kConcurrentExecution, msg); }
  static Status NotReady(const std::string& msg) { return Status(kNotReady, msg); }
  static Status NetworkError(const std::string& msg) { return Status(kNetworkError, msg);}
  static Status Timeout(const std::string& msg) { return Status(kTimeout, msg);}
  static Status ServerFuncNotFound(const std::string& msg) { return Status(kServerFuncNotFound, msg);}
  static Status ServerSerializeFailed(const std::string& msg) { return Status(kServerSerializeFailed, msg);}
  static Status ServerDeserializeFailed(const std::string& msg) { return Status(kServerDeserializeFailed, msg);}
  static Status ClientSerializeFailed(const std::string& msg) { return Status(kClientSerializeFailed, msg);}
  static Status ClientDeserializeFailed(const std::string& msg) { return Status(kClientDeserializeFailed, msg);}
  static Status FileQueueNeedWait(const std::string& msg) { return Status(kFileQueueNeedWait, msg);}
  static Status Unknown(const std::string& msg) { return Status(kUnknown, msg); }
  

 private:
  friend class ps::serializer::SerializeHelper;
  struct State {
    ErrorCode code;
    std::string msg;
  };

  State* state_;
};

}

#define PS_CHECK_STATUS(STATUS)			\
  do {						\
    ps::Status __st__ = STATUS;			\
    if (!__st__.IsOk()) {			\
      return __st__;				\
    }						\
  } while (0)

#define PS_CHECK_BOOL(BOOL, STATUS) \
  do {                              \
    if (!(BOOL)) {                  \
      return STATUS;                \
    }                               \
  } while (0)

#endif

