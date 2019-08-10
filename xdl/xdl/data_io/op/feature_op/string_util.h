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


#pragma once

#include <string>
#include <vector>

namespace xdl {
namespace io {

class StringUtil {
 public:
  static std::string &Trim(std::string &str);
  static std::string Trim(const std::string &str);
  static void Split(const std::string &src,
                    std::vector<std::string> &dst,
                    const std::string &separator);
  static void SplitExclude(const std::string &src, std::vector<std::string> &dst,
                           char separator, char left_exclude, char right_exclude);
  static void SplitFirst(const std::string &src,
                         std::vector<std::string> &dst,
                         const std::string &separator);

 private:
  static const std::string space_;
};

}  // namespace io
}  // namespace xdl