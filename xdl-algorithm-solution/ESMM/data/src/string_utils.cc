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

#include "string_utils.h"

#include <vector>
#include <string>
#include <mutex>
#include <iostream>

using namespace std;

namespace xdl {

class StringStreamPool {
 public:
  StringStreamPool() {
    _ssVec.reserve(8);
  }
  ~StringStreamPool() {
    for (size_t i = 0; i < _ssVec.size(); ++i) {
      delete _ssVec[i];
    }
  }
  stringstream* getStringStream() {
    stringstream* ret = NULL;
    {
      std::unique_lock<mutex> lock(_lock);
      if (!_ssVec.empty()) {
        ret = _ssVec.back();
        _ssVec.pop_back();
      }
    }
    if (ret == NULL) {
      ret = new stringstream();
    }
    return ret;
  }
  void putStringStream(stringstream* ss) {
    ss->clear();
    std::unique_lock<mutex> lock(_lock);
    _ssVec.push_back(ss);
  }
 private:
  std::mutex _lock;
  vector<stringstream *> _ssVec; 
};

static const size_t POOL_SIZE = 31;
static StringStreamPool sPool[POOL_SIZE];

stringstream* StringUtils::getStringStream() {
  size_t offset = pthread_self() % POOL_SIZE;
  return sPool[offset].getStringStream();
}

void StringUtils::putStringStream(stringstream* ss) {
  size_t offset = pthread_self() % POOL_SIZE;
  sPool[offset].putStringStream(ss);
}

bool StringUtils::strToInt8(const char* str, int8_t& value) {
  int32_t v32 = 0;
  bool ret = strToInt32(str, v32);
  value = (int8_t)v32;
  return ret && v32 >= INT8_MIN && v32 <= INT8_MAX;
}

bool StringUtils::strToUInt8(const char* str, uint8_t& value) {
  uint32_t v32 = 0;
  bool ret = strToUInt32(str, v32);
  value = (uint8_t)v32;
  return ret && v32 <= UINT8_MAX;
}

bool StringUtils::strToInt16(const char* str, int16_t& value) {
  int32_t v32 = 0;
  bool ret = strToInt32(str, v32);
  value = (int16_t)v32;    
  return ret && v32 >= INT16_MIN && v32 <= INT16_MAX;
}

bool StringUtils::strToUInt16(const char* str, uint16_t& value) {
  uint32_t v32 = 0;
  bool ret = strToUInt32(str, v32);
  value = (uint16_t)v32;
  return ret && v32 <= UINT16_MAX;
}

bool StringUtils::strToInt32(const char* str, int32_t& value) {
  if (NULL == str || *str == 0) {
    return false;
  }

  char* endPtr = NULL;
  errno = 0;
# if __WORDSIZE == 64
  int64_t value64 = strtol(str, &endPtr, 10);
  if (value64 < INT32_MIN || value64 > INT32_MAX) {
    return false;
  }

  value = (int32_t)value64;
# else
  value = (int32_t)strtol(str, &endPtr, 10);
# endif

  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }

  return false;
}

bool StringUtils::strToUInt32(const char* str, uint32_t& value) {
  if (NULL == str || *str == 0 || *str == '-') {
    return false;
  }

  char* endPtr = NULL;
  errno = 0;
# if __WORDSIZE == 64
  uint64_t value64 = strtoul(str, &endPtr, 10);
  if (value64 > UINT32_MAX) {
    return false;
  }

  value = (int32_t)value64;
# else
  value = (uint32_t)strtoul(str, &endPtr, 10);
# endif

  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }

  return false;
}

bool StringUtils::strToUInt64(const char* str, uint64_t& value) {
  if (NULL == str || *str == 0 || *str == '-') {
    return false;
  }

  char* endPtr = NULL;
  errno = 0;
  value = (uint64_t)strtoull(str, &endPtr, 10);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }

  return false;
}

bool StringUtils::strToInt64(const char* str, int64_t& value) {
  if (NULL == str || *str == 0) {
    return false;
  }

  char* endPtr = NULL;
  errno = 0;
  value = (int64_t)strtoll(str, &endPtr, 10);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }

  return false;
}

bool StringUtils::strToFloat(const char* str, float& value) 
{
  if (NULL == str || *str == 0) 
  {
    return false;
  }
  errno = 0;
  char* endPtr = NULL;
  value = strtof(str, &endPtr);
  if (errno == 0 && endPtr && *endPtr == 0) 
  {
    return true;
  }
  return false;
}

bool StringUtils::strToDouble(const char* str, double& value) 
{
  if (NULL == str || *str == 0) 
  {
    return false;
  }
  errno = 0;
  char* endPtr = NULL;
  value = strtod(str, &endPtr);
  if (errno == 0 && endPtr && *endPtr == 0) 
  {
    return true;
  }
  return false;
}

std::vector<std::string> StringUtils::split(
    const std::string& text, 
    const std::string &sepStr, 
    bool ignoreEmpty) {
  std::vector<std::string> vec;
  std::string str(text);
  std::string sep(sepStr);
  size_t n = 0, old = 0;
  while (n != std::string::npos) {
    n = str.find(sep,n);
    if (n != std::string::npos) {
      if (!ignoreEmpty || n != old) 
        vec.push_back(str.substr(old, n-old));
      n += sep.length();
      old = n;
    }
  }

  if (!ignoreEmpty || old < str.length()) {
    vec.push_back(str.substr(old, str.length() - old));
  }

  return vec;
}

bool StringUtils::GetValueFromMap(
    const map<string, string>& params, 
    const string& key,
    string* value) {
  auto it = params.find(key);
  if (it != params.end()) {
    *value = it->second;
  } else {
    std::cerr << "key:" << key << " not found in map..." << std::endl;
    return false;
  }

  return true;
}

} // xdl
