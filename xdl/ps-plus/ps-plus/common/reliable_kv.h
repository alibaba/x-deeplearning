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

#ifndef PS_PLUS_COMMON_RELIABLE_KV_H_
#define PS_PLUS_COMMON_RELIABLE_KV_H_

#include "ps-plus/common/plugin.h"
#include "ps-plus/common/status.h"

namespace ps {

class ReliableKV {
 public:
  virtual Status Read(const std::string& addr, std::string* val, int retry) = 0;
  virtual Status Write(const std::string& addr, const std::string& val, int retry) = 0;
  virtual ~ReliableKV() {}
  static Status ReadAny(const std::string& addr, std::string* val, int retry = 3);
  static Status WriteAny(const std::string& addr, const std::string& val, int retry = 3);
};

};

#endif

