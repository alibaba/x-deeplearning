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

#ifndef TDM_SERVING_UTIL_TO_STRING_H_
#define TDM_SERVING_UTIL_TO_STRING_H_

#include <stdio.h>
#include <string>
#include <sstream>
#include <iterator>
#include <algorithm>

namespace tdm_serving {
namespace util {

#define TO_STRING(format, val) \
  do { \
    char buf[256]; \
    int32_t n = snprintf(buf, sizeof (buf), format, (val)); \
    return std::string(buf, n); \
  } while (0)

inline std::string ToString(const std::string& val) { return val; }

inline std::string ToString(int32_t val) { TO_STRING("%d", val); }
inline std::string ToString(uint32_t val) { TO_STRING("%u", val); }

inline std::string ToString(int64_t val) {
  TO_STRING("%lld", (long long int)val);
}

inline std::string ToString(uint64_t val) {
  TO_STRING("%llu", (unsigned long long int)val);
}

inline std::string ToString(float val) { TO_STRING("%f", val); }
inline std::string ToString(double val) { TO_STRING("%f", val); }
#undef TO_STRING

template<typename C>
std::string ToString(const C& c, const char* delimiter) {
  std::stringstream ss;
  std::ostream_iterator<typename C::value_type> out(ss, delimiter);
  std::copy(c.begin(), c.end(), out);
  std::string str = ss.str();
  return str.substr(0, str.size() - 1);  // -1 for tail delimiter
}

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_TO_STRING_H_
