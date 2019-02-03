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

#ifndef TDM_SERVING_UTIL_STR_UTIL_H_
#define TDM_SERVING_UTIL_STR_UTIL_H_

#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include "common/common_def.h"
#include "util/to_string.h"

namespace tdm_serving {
namespace util {

class StrUtil {
 public:
  StrUtil();
  ~StrUtil();

 public:
  // cast string to any type
  template <typename Type>
  static bool StrConvert(const char* str, Type* value);

  // cast int string
  static bool IntToString(int val, char *buf, int max_len);

  // cast long string
  static bool LongToString(long val, char *buf, int max_len);

  // split
  static void Split(
    const std::string &str, const std::string &sep,
    bool ignoreEmpty, std::vector<std::string> *vec);

  static void Split(
    const std::string &str, const char& sep,
    bool ignoreEmpty, std::vector<std::string> *vec);

  static void SplitBySpace(const std::string &str,
    bool ignoreEmpty, std::vector<std::string> *vec);

  static void Split(char* str, char sep,
    bool ignoreEmpty, std::vector<char*> *vec);

  static void StringReplace(std::string& strBig,
    const std::string& strsrc, const std::string& strdst);

  static void SplitAll(std::vector<std::string>& vs,
    const std::string& line, char dmt);

  static std::string Trim(std::string& str);

  static char* Trim(char* str);

  static bool SubStr(const std::string& str,
    const std::string& start_tag, const std::string& end_tag,
    std::string* sub_str);

 private:
  DISALLOW_COPY_AND_ASSIGN(StrUtil);
};

template <>
bool StrUtil::StrConvert<bool>(const char* str, bool* value);

template <>
bool StrUtil::StrConvert<int32_t>(const char* str, int32_t* value);

template <>
bool StrUtil::StrConvert<uint32_t>(const char* str, uint32_t* value);

template <>
bool StrUtil::StrConvert<int64_t>(const char* str, int64_t* value);

template <>
bool StrUtil::StrConvert<uint64_t>(const char* str, uint64_t* value);

template <>
bool StrUtil::StrConvert<float>(const char* str, float* value);

template <>
bool StrUtil::StrConvert<double>(const char* str, double* value);

template <>
bool StrUtil::StrConvert<std::string>(const char* str, std::string* value);

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_STR_UTIL_H_
