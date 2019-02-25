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

#include <errno.h>
#include <iconv.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "util/str_util.h"

namespace tdm_serving {
namespace util {

StrUtil::StrUtil() {
}

StrUtil::~StrUtil() {
}

template <>
bool StrUtil::StrConvert<bool>(const char* str, bool* value) {
  if (str == NULL || value == NULL) {
    return false;
  }
  if (strcasecmp(str, "true") == 0
      || strcasecmp(str, "on") == 0
      || strcmp(str, "1") == 0) {
    *value = true;
  } else if (strcasecmp(str, "false") == 0
      || strcasecmp(str, "off") == 0
      || strcmp(str, "0") == 0) {
    *value = false;
  } else {
    return false;
  }
  return true;
}

template <>
bool StrUtil::StrConvert<int32_t>(const char* str, int32_t* value) {
  if (str == NULL || value == NULL ||str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtol(str, &endptr, 10);
  return (*endptr != '\0') ? false : true;
}

template <>
bool StrUtil::StrConvert<uint32_t>(const char* str, uint32_t* value) {
  if (str == NULL || value == NULL || str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtol(str, &endptr, 10);
  return (*endptr != '\0') ? false : true;
}

template <>
bool StrUtil::StrConvert<int64_t>(const char* str, int64_t* value) {
  if (str == NULL || value == NULL ||str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtoll(str, &endptr, 10);
  return (*endptr != '\0') ? false : true;
}

template <>
bool StrUtil::StrConvert<uint64_t>(const char* str, uint64_t* value) {
  if (str == NULL || value == NULL || str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtoll(str, &endptr, 10);
  return (*endptr != '\0') ? false : true;
}

template <>
bool StrUtil::StrConvert<std::string>(const char* str, std::string* value) {
  if (str == NULL || value == NULL) {
    return false;
  }
  value->assign(str);
  return true;
}

template <>
bool StrUtil::StrConvert<float>(const char* str, float* value) {
  if (str == NULL || value == NULL ||str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtof(str, &endptr);
  return (*endptr != '\0') ? false : true;
}

template <>
bool StrUtil::StrConvert<double>(const char* str, double* value) {
  if (str == NULL || value == NULL ||str[0] == '\0') {
    return false;
  }
  char* endptr = NULL;
  *value = strtod(str, &endptr);
  return (*endptr != '\0') ? false : true;
}

bool StrUtil::IntToString(int val, char *buf, int max_len) {
  if (snprintf(buf, max_len, "%d", val) <= 0) {
    return false;
  }
  return true;
}

bool StrUtil::LongToString(long val, char *buf, int max_len) {
  if (snprintf(buf, max_len, "%ld", val) <= 0) {
    return false;
  }
  return true;
}

void StrUtil::Split(const std::string &str, const std::string &sep,
    bool ignoreEmpty, std::vector<std::string> *vec) {
  assert(vec != NULL);
  vec->clear();
  size_t old = 0;
  size_t n = str.find(sep);
  while (n != std::string::npos) {
    if (!ignoreEmpty || n != old) {
      vec->push_back(str.substr(old, n - old));
    }
    n += sep.length();
    old = n;
    n = str.find(sep, n);
  }
  if (!ignoreEmpty || old < str.length()) {
    vec->push_back(str.substr(old));
  }
}

void StrUtil::Split(const std::string &str, const char &sep,
    bool ignoreEmpty, std::vector<std::string> *vec) {
  Split(str, std::string(1, sep), ignoreEmpty, vec);
}

void StrUtil::SplitBySpace(const std::string &str,
    bool ignoreEmpty, std::vector<std::string> *vec) {
  if (vec == NULL) {
    return;
  }
  vec->clear();
  size_t old = 0;
  while (true) {
    size_t curr = old;
    while (curr < str.length() && !isspace(str[curr])) {
      ++curr;
    }
    if (!ignoreEmpty || curr - old > 0) {
      vec->push_back(str.substr(old, curr - old));
    }
    if (curr >= str.length()) {
      break;
    }
    old = curr + 1;
  }
}

void StrUtil::Split(char* str, char sep,
    bool ignoreEmpty, std::vector<char*> *vec) {
  if (str == NULL || vec == NULL) {
    return;
  }
  vec->clear();
  char* b = str;
  char* p = b;
  while (p != NULL && *p !='\0') {
    p = strchr(b, sep);
    if (p != NULL) {
      if (p != b || !ignoreEmpty) {
        (*p) = 0;
        vec->push_back(b);
      }
      p += 1;
      b = p;
    } else {
      vec->push_back(b);
    }
  }
  if (p == b && !ignoreEmpty) {
    vec->push_back(p);
  }
}

void StrUtil::StringReplace(std::string & strBig,
    const std::string & strsrc, const std::string &strdst) {
  std::string::size_type pos = 0;
  std::string::size_type srclen = strsrc.size();
  std::string::size_type dstlen = strdst.size();
  while ((pos=strBig.find(strsrc, pos)) != std::string::npos) {
    strBig.replace(pos, srclen, strdst);
    pos += dstlen;
  }
}

void StrUtil::SplitAll(std::vector<std::string>& vs,
    const std::string& line, char dmt = '\t') {
  std::string::size_type p = 0;
  std::string::size_type q;
  vs.clear();
  for (;;) {
    q = line.find(dmt, p);
    std::string str = line.substr(p, q - p);
    Trim(str);
    vs.push_back(str);
    if (q == std::string::npos) break;
    p = q+1;
  }
}

std::string StrUtil::Trim(std::string& str) {
  std::string::size_type p = str.find_first_not_of(" \t\r\n");
  if (p == std::string::npos) {
    str = "";
    return str;
  }
  std::string::size_type q = str.find_last_not_of(" \t\r\n");
  str = str.substr(p, q-p+1);
  return str;
}

char* StrUtil::Trim(char* str) {
  if (*str == '\0') return str;

  char* tail = str + strlen(str) - 1;
  while ((tail > str) && (isspace(*tail) != 0)) --tail;
  *(tail+1) = '\0';

  while (*str != '\0' && (isspace(*str) != 0)) ++str;
  return str;
}

bool StrUtil::SubStr(const std::string& str,
    const std::string& start_tag, const std::string& end_tag,
    std::string* sub_str) {
  size_t start = 0;
  size_t end = 0;
  size_t len = start_tag.length();

  if ((start = str.find(start_tag)) != std::string::npos &&
      (end = str.find(end_tag, start)) != std::string::npos) {
    *sub_str = str.substr(start + len, end - start - len);
    return true;
  }
  return false;
}

}  // namespace util
}  // namespace tdm_serving

