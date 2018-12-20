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

#ifndef PS_SERVICE_SEASTAR_LIB_SEASTAR_STATUS_H_
#define PS_SERVICE_SEASTAR_LIB_SEASTAR_STATUS_H_

#include <memory>
#include <string>
#include <algorithm>

namespace ps {
namespace service {
namespace seastar {

class SeastarStatus {
 public:
  enum ErrorCode {
    kOk = 0,
    kNetworkError,
    kTimeout,
    kServerFuncNotFound,
    kServerSerializeFailed,
    kServerDeserializeFailed,
    kClientSerializeFailed,
    kClientDeserializeFailed,
  };

  SeastarStatus(): 
    code_(kOk) {}
  SeastarStatus(ErrorCode code) : 
    code_(code) {}

  std::string ToString() const {
    switch (code_) {
    case kOk:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "success";
      break;
    case kNetworkError:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "network error";
      break;      
    case kTimeout:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "timeout";
      break;      
    case kServerFuncNotFound:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "server func not found";
      break;      
    case kServerSerializeFailed:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "server serialize failed";
      break;      
    case kServerDeserializeFailed:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "server deserialize failed";
      break;      
    case kClientSerializeFailed:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "client serialize failed";
      break;      
    case kClientDeserializeFailed:
      return "ErrorCode[" + std::to_string(code_) + "]:" + "client serialize failed";
      break;      
    default:
      return "unknown error";
      break;
    }
  }

  ErrorCode Code() const {
    return code_;
  }

  bool Success() const {
    return code_ == kOk;
  }

  void* Data() {
    return &code_;
  }

  int Size() {
    return sizeof(ErrorCode);
  }

  bool operator==(const SeastarStatus& rhs) const {
    return code_ == rhs.code_;
  }

  static SeastarStatus Ok() { return SeastarStatus(kOk); }
  static SeastarStatus NetworkError() { return SeastarStatus(kNetworkError); }
  static SeastarStatus Timeout() { return SeastarStatus(kTimeout); }
  static SeastarStatus ServerFuncNotFound() { return SeastarStatus(kServerFuncNotFound); }
  static SeastarStatus ServerSerializeFailed() { return SeastarStatus(kServerSerializeFailed); }
  static SeastarStatus ServerDeserializeFailed() { return SeastarStatus(kServerDeserializeFailed); }
  static SeastarStatus ClientSerializeFailed() { return SeastarStatus(kClientSerializeFailed); }
  static SeastarStatus ClientDeserializeFailed() { return SeastarStatus(kClientDeserializeFailed); }

 private:
  ErrorCode code_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_SEASTAR_STATUS_H_

