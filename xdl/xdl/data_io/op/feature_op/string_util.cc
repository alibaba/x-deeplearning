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


#include "xdl/data_io/op/feature_op/string_util.h"

#include <xdl/core/utils/logging.h>

namespace xdl {
namespace io {

const std::string StringUtil::space_ = " 　";

std::string &StringUtil::Trim(std::string &str) {
  if (str.empty())  return str;
  str.erase(0, str.find_first_not_of(space_));
  str.erase(str.find_last_not_of(space_) + 1);
  return str;
}

std::string StringUtil::Trim(const std::string &str) {
  if (str.empty())  return str;
  std::string ret = std::move(str);
  ret.erase(0, ret.find_first_not_of(space_));
  ret.erase(ret.find_last_not_of(space_) + 1);
  return ret;
}

void StringUtil::Split(const std::string &src,
                       std::vector<std::string> &dst,
                       const std::string &separator) {
  if (src.empty() || separator.empty())  return;
  size_t offset = 0;
  for (size_t pos = 0; (pos = src.find_first_of(separator, offset)) != std::string::npos; offset = pos + 1) {
    const std::string &sub = Trim(src.substr(offset, pos - offset));
    if (sub.length() > 0)  dst.push_back(sub);
  }
  const std::string &sub = Trim(src.substr(offset, src.length() - offset));
  if (sub.length() > 0)  dst.push_back(sub);
}

void StringUtil::SplitExclude(const std::string &src, std::vector<std::string> &dst,
                              char separator, char left_exclude, char right_exclude) {
  if (src.empty())  return;
  int sub_level = 0;
  for (size_t offset = 0, pos = 0; pos <= src.size(); ++pos) {
    if (pos == src.size() || (src[pos] == separator && sub_level == 0)) {
      const std::string &sub = Trim(src.substr(offset, pos - offset));
      if (sub.length() > 0)  dst.push_back(sub);
      offset = pos + 1;
    } else if (src[pos] == left_exclude) {
      ++sub_level;
    } else if (src[pos] == right_exclude) {
      --sub_level;
      XDL_CHECK(sub_level >= 0);
    }
  }
}

void StringUtil::SplitFirst(const std::string &src,
                            std::vector<std::string> &dst,
                            const std::string &separator) {
  if (src.empty() || separator.empty())  return;
  size_t pos = src.find_first_of(separator);
  if (pos == std::string::npos) {
    dst.push_back(Trim(src));
  } else {
    dst.push_back(Trim(src.substr(0, pos++)));
    dst.push_back(Trim(src.substr(pos, src.length() - pos)));
  }
}

}  // namespace io
}  // namespace xdl
